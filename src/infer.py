import os, glob, argparse, torch

from config import config
from transformerModel import Transformer
from dataloader import en_tok, fr_tok, SRC_BOS_IDX, SRC_EOS_IDX, TGT_BOS_IDX, TGT_EOS_IDX, MAX_LEN

def _resolve_ckpt(path_or_dir: str) -> str:
    if os.path.isfile(path_or_dir):
        return path_or_dir
    pts = sorted(glob.glob(os.path.join(path_or_dir, "*.pt")), key=os.path.getmtime, reverse=True)
    if not pts:
        raise FileNotFoundError(f"No .pt files in {path_or_dir}")
    return pts[0]

def _build_model(hp=None) -> Transformer:
    SRC_V, TGT_V = en_tok.get_vocab_size(), fr_tok.get_vocab_size()
    d_model    = hp.get("d_model", config.d_model)       if hp else config.d_model
    n_heads    = hp.get("n_heads", config.num_heads)     if hp else config.num_heads
    num_layers = hp.get("num_layers", config.num_layers) if hp else config.num_layers
    d_ff       = hp.get("d_ff", config.d_ff)             if hp else config.d_ff
    max_len    = hp.get("max_len", MAX_LEN)              if hp else MAX_LEN
    return Transformer(SRC_V, TGT_V, d_model=d_model, n_heads=n_heads,
                       num_layers=num_layers, d_ff=d_ff, max_len=max_len).to(config.device)

def _load_model(ckpt_path: str) -> Transformer:
    obj = torch.load(ckpt_path, map_location=config.device)
    hp = obj.get("hparams") if isinstance(obj, dict) else None
    model = _build_model(hp)
    state = obj["model_state"] if isinstance(obj, dict) and "model_state" in obj else obj
    model.load_state_dict(state)
    model.eval()
    return model

def _encode_src(sentence: str) -> torch.Tensor:
    ids = en_tok.encode(str(sentence)).ids
    max_inner = max(2, getattr(config, "max_seq_length", MAX_LEN) - 2)
    ids = [SRC_BOS_IDX] + ids[:max_inner] + [SRC_EOS_IDX]
    return torch.tensor(ids, dtype=torch.long, device=config.device).unsqueeze(1)  # [T,1]

def _ids_to_text(ids1d: torch.Tensor) -> str:
    toks = []
    for i in ids1d.tolist():
        if i == TGT_BOS_IDX: continue
        if i == TGT_EOS_IDX: break
        toks.append(fr_tok.id_to_token(int(i)))
    return " ".join(toks)

@torch.no_grad()
def greedy_decode(model: Transformer, src: torch.Tensor, max_len: int) -> torch.Tensor:
    tgt = torch.tensor([[TGT_BOS_IDX]], dtype=torch.long, device=src.device)  # [1,1]
    for _ in range(max_len - 1):
        out = model(src, tgt)                         # [Tcur,1,V]
        next_id = int(out[-1, 0].argmax(-1).item())
        tgt = torch.cat([tgt, torch.tensor([[next_id]], device=src.device)], dim=0)
        if next_id == TGT_EOS_IDX: break
    return tgt[:, 0]  # [T]

@torch.no_grad()
def beam_search_decode(model: Transformer, src: torch.Tensor, max_len: int, beam_size: int) -> torch.Tensor:
    beams = [(torch.tensor([[TGT_BOS_IDX]], device=src.device, dtype=torch.long), 0.0, False)]
    for _ in range(max_len - 1):
        cand = []
        for ids, score, done in beams:
            if done:
                cand.append((ids, score, True)); continue
            out = model(src, ids)
            logp = torch.log_softmax(out[-1, 0], dim=-1)
            topk = torch.topk(logp, k=beam_size)
            for lp, tok in zip(topk.values.tolist(), topk.indices.tolist()):
                if ids.size(0) == 1 and tok in (TGT_BOS_IDX, TGT_EOS_IDX):  # avoid empty output
                    continue
                tok_t = torch.tensor([[tok]], device=src.device, dtype=torch.long)
                ids_new = torch.cat([ids, tok_t], dim=0)
                cand.append((ids_new, score + lp, tok == TGT_EOS_IDX))
        if not cand: break
        cand.sort(key=lambda x: x[1], reverse=True)
        beams = cand[:beam_size]
        if all(d for _, _, d in beams): break

    def length_norm(score, length, alpha=0.7):
        return score / ((5 + length) ** alpha / (5 + 1) ** alpha)
    best = max(beams, key=lambda x: length_norm(x[1], x[0].size(0)))
    return best[0][:, 0]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None, help="Path to .pt or a directory (defaults to models/)")
    ap.add_argument("--src", required=True, help="English sentence to translate")
    ap.add_argument("--beam", type=int, default=None, help="Beam size (default: config.beam_size)")
    ap.add_argument("--max_len", type=int, default=None, help="Max gen length (default: config.max_gen_len)")
    args = ap.parse_args()

    ckpt_path = _resolve_ckpt(args.ckpt or getattr(config, "MODELS_DIR", "models"))
    beam_size = args.beam if args.beam is not None else getattr(config, "beam_size", 1)
    max_len = args.max_len if args.max_len is not None else getattr(config, "max_gen_len", MAX_LEN)

    print(f"Device: {config.device} | Checkpoint: {ckpt_path}")
    model = _load_model(ckpt_path)
    src = _encode_src(args.src)

    out_ids = beam_search_decode(model, src, max_len, beam_size) if beam_size and beam_size > 1 \
              else greedy_decode(model, src, max_len)
    text = _ids_to_text(out_ids)
    print(text if text.strip() else "<empty>")

if __name__ == "__main__":
    main()
