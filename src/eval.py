import os, glob, math, argparse, torch
from tqdm import tqdm

from config import config
from transformerModel import Transformer
from dataloader import (
    TranslationDataset, collate_fn, en_tok, fr_tok,
    SRC_PAD_IDX, SRC_BOS_IDX, SRC_EOS_IDX,
    TGT_PAD_IDX, TGT_BOS_IDX, TGT_EOS_IDX, MAX_LEN
)

def _resolve_ckpt(path_or_dir: str) -> str:
    if os.path.isfile(path_or_dir):
        return path_or_dir
    pts = sorted(glob.glob(os.path.join(path_or_dir, "*.pt")), key=os.path.getmtime, reverse=True)
    if not pts:
        raise FileNotFoundError(f"No .pt files in {path_or_dir}")
    return pts[0]

def _build_model(hp=None):
    SRC_V, TGT_V = en_tok.get_vocab_size(), fr_tok.get_vocab_size()
    d_model    = hp.get("d_model", config.d_model)       if hp else config.d_model
    n_heads    = hp.get("n_heads", config.num_heads)     if hp else config.num_heads
    num_layers = hp.get("num_layers", config.num_layers) if hp else config.num_layers
    d_ff       = hp.get("d_ff", config.d_ff)             if hp else config.d_ff
    max_len    = hp.get("max_len", MAX_LEN)              if hp else MAX_LEN
    return Transformer(SRC_V, TGT_V, d_model=d_model, n_heads=n_heads,
                       num_layers=num_layers, d_ff=d_ff, max_len=max_len).to(config.device)

def _load_model(ckpt_path: str):
    obj = torch.load(ckpt_path, map_location=config.device)
    hp = obj.get("hparams") if isinstance(obj, dict) else None
    model = _build_model(hp)
    state = obj["model_state"] if isinstance(obj, dict) and "model_state" in obj else obj
    model.load_state_dict(state)
    model.eval()
    return model

def _ids_to_text(ids1d: torch.Tensor) -> str:
    toks = []
    for i in ids1d.tolist():
        if i == TGT_BOS_IDX: continue
        if i == TGT_EOS_IDX: break
        toks.append(fr_tok.id_to_token(int(i)))
    return " ".join(toks)

@torch.no_grad()
def _greedy(model, src, max_len):
    tgt = torch.tensor([[TGT_BOS_IDX]], dtype=torch.long, device=src.device)
    for _ in range(max_len - 1):
        out = model(src, tgt)                 # [Tcur,1,V]
        nid = int(out[-1, 0].argmax(-1).item())
        tgt = torch.cat([tgt, torch.tensor([[nid]], device=src.device)], dim=0)
        if nid == TGT_EOS_IDX: break
    return tgt[:, 0]

def main():
    try:
        import sacrebleu
        use_sacre = True
    except Exception:
        from nltk.translate.bleu_score import corpus_bleu
        use_sacre = False

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None, help="Path to .pt or a directory (defaults to models dir)")
    ap.add_argument("--limit", type=int, default=None, help="Evaluate first N examples")
    ap.add_argument("--out", default="test_translations.txt")
    args = ap.parse_args()

    ckpt_dir_or_file = args.ckpt or getattr(config, "MODELS_DIR", "models")
    ckpt = _resolve_ckpt(ckpt_dir_or_file)
    print("Device:", config.device, "| Using checkpoint:", ckpt)

    model = _load_model(ckpt)

    ds = TranslationDataset("test")
    if args.limit is not None:
        from torch.utils.data import Subset
        ds = Subset(ds, list(range(min(args.limit, len(ds)))))

    dl = torch.utils.data.DataLoader(
        ds, batch_size=1, shuffle=False, collate_fn=collate_fn,
        num_workers=2, pin_memory=(config.device.type == "cuda")
    )

    crit = torch.nn.CrossEntropyLoss(ignore_index=TGT_PAD_IDX, label_smoothing=getattr(config, "label_smoothing", 0.0))
    refs, hyps = [], []
    loss_sum, steps = 0.0, 0

    with torch.no_grad():
        for src, tgt in tqdm(dl, desc="Eval test"):
            src, tgt = src.to(config.device), tgt.to(config.device)

            # teacher-forced loss (ignore target PAD)
            out = model(src, tgt[:-1, :])  # [T-1,1,V]
            V = out.shape[-1]
            loss_sum += float(crit(out.reshape(-1, V), tgt[1:, :].reshape(-1)).item())
            steps += 1

            # greedy generation for BLEU/chrF
            gen = _greedy(model, src, max_len=getattr(config, "max_gen_len", MAX_LEN))
            hyps.append(_ids_to_text(gen))
            refs.append(_ids_to_text(tgt[:, 0]))

    val_loss = loss_sum / max(1, steps)
    ppl = math.exp(val_loss) if val_loss < 20 else float("inf")

    if use_sacre:
        bleu = sacrebleu.corpus_bleu(hyps, [refs]).score
        chrf = sacrebleu.corpus_chrf(hyps, [refs]).score
    else:
        from nltk.translate.bleu_score import corpus_bleu
        bleu = corpus_bleu([[r.split()] for r in refs], [h.split() for h in hyps]) * 100
        chrf = None

    print(f"Test Loss: {val_loss:.3f} | PPL: {ppl:.2f} | BLEU: {bleu:.2f}" +
          (f" | chrF2: {chrf:.2f}" if chrf is not None else ""))

    with open(args.out, "w", encoding="utf-8") as f:
        for h in hyps:
            f.write(h.strip() + "\n")
    print(f"Wrote {len(hyps)} translations to {args.out}")

if __name__ == "__main__":
    main()
