import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu

try:
    import sacrebleu
except Exception:
    sacrebleu = None

from config import config
from transformerModel import Transformer
from dataloader import (TranslationDataset, collate_fn, en_tok, fr_tok,
    SRC_PAD_IDX, SRC_BOS_IDX, SRC_EOS_IDX, TGT_PAD_IDX, TGT_BOS_IDX, TGT_EOS_IDX, MAX_LEN)


print("Device is:", config.device)

# ================== Data ==================
train_dataset = TranslationDataset("train")
train_dataloader = DataLoader(
    train_dataset, batch_size=config.trainBatchSize, shuffle=True, collate_fn=collate_fn
)
val_dataset = TranslationDataset("val")
val_dataloader = DataLoader(
    val_dataset, batch_size=config.valBatchSize, shuffle=False, collate_fn=collate_fn
)

# ================== Model ==================
SRC_VOCAB_SIZE = en_tok.get_vocab_size()
TGT_VOCAB_SIZE = fr_tok.get_vocab_size()
print("Vocab sizes:", SRC_VOCAB_SIZE, TGT_VOCAB_SIZE)

model = Transformer(
    SRC_VOCAB_SIZE, TGT_VOCAB_SIZE,
    d_model=config.d_model, n_heads=config.num_heads,
    num_layers=config.num_layers, d_ff=config.d_ff, max_len=MAX_LEN
).to(config.device)

optimizer = optim.Adam(model.parameters(), lr=config.lr, betas=(0.9, 0.98), eps=1e-9)

# ---- Noam warmup scheduler (uses config.warmup_steps) ----
def noam_lambda(step):
    step = max(step, 1)
    return (config.d_model ** -0.5) * min(step ** -0.5, step * (config.warmup_steps ** -1.5))
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=noam_lambda)

# Loss with label smoothing (uses config.label_smoothing)
criterion = nn.CrossEntropyLoss(ignore_index=TGT_PAD_IDX, label_smoothing=config.label_smoothing)


# Seed + AMP (uses config.seed, config.mixed_precision)
torch.manual_seed(getattr(config, "seed", 42))
use_amp = (getattr(config, "mixed_precision", True) and config.device.type == "cuda")
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

best_val = float("inf")  # <--- track best validation loss

for epoch in range(config.num_epochs):
    model.train()
    epoch_loss = 0.0
    progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}")
    for i, (src, tgt) in enumerate(progress_bar):
        src, tgt = src.to(config.device), tgt.to(config.device)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(src, tgt[:-1, :])           # [T-1,B,V]
            V = logits.shape[-1]
            loss = criterion(logits.reshape(-1, V), tgt[1:, :].reshape(-1))
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()  # Noam per-batch
        epoch_loss += float(loss.item())
        progress_bar.set_postfix(batch_loss=float(loss.item()), avg_loss=epoch_loss / (i + 1))

    # -------- Validation --------
    model.eval()
    val_references, val_hypotheses = [], []
    val_loss, val_steps = 0.0, 0
    correct_tokens, total_tokens = 0, 0  # streaming accuracy

    with torch.no_grad():
        for j, (src, tgt) in enumerate(val_dataloader):
            src, tgt = src.to(config.device), tgt.to(config.device)

            out = model(src, tgt[:-1, :])                 # [T-1,B,V]
            V = out.shape[-1]
            batch_loss = criterion(out.reshape(-1, V), tgt[1:, :].reshape(-1)).item()
            val_loss += float(batch_loss); val_steps += 1

            # token-accuracy (teacher-forced) — streaming counters
            pred_ids = out.argmax(-1)                     # [T-1,B]
            mask = (tgt[1:, :] != TGT_PAD_IDX)            # [T-1,B]
            correct_tokens += (pred_ids[mask] == tgt[1:, :][mask]).sum().item()
            total_tokens   += mask.sum().item()

            # texts for BLEU/chrF (target-side IDs!)
            for b in range(tgt.size(1)):
                ref = [fr_tok.id_to_token(int(idx)) for idx in tgt[1:, b]
                    if int(idx) not in (TGT_PAD_IDX, TGT_EOS_IDX)]
                hyp = [fr_tok.id_to_token(int(idx)) for idx in pred_ids[:, b]
                    if int(idx) not in (TGT_PAD_IDX, TGT_EOS_IDX)]
                val_references.append([ref])
                val_hypotheses.append(hyp)

    val_loss_avg = (val_loss / max(val_steps, 1))
    val_ppl = math.exp(val_loss_avg) if val_loss_avg < 20 else float("inf")
    val_token_acc = (correct_tokens / total_tokens) if total_tokens else 0.0

    # BLEU/chrF (prefer sacrebleu; NLTK with smoothing fallback)
    if sacrebleu:
        refs = [" ".join(r[0]) for r in val_references]
        hyps = [" ".join(h)    for h in val_hypotheses]
        val_bleu = sacrebleu.corpus_bleu(hyps, [refs]).score / 100.0  # keep 0..1 scale
        val_chrf = sacrebleu.corpus_chrf(hyps, [refs]).score
    else:
        from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
        sm = SmoothingFunction().method3
        val_bleu = corpus_bleu(val_references, val_hypotheses, smoothing_function=sm)
        val_chrf = None


    print(
        f"Epoch {epoch+1}, "
        f"Train Loss: {epoch_loss/len(train_dataloader):.3f}, "
        f"Val Loss: {val_loss_avg:.3f}, "
        f"Val PPL: {val_ppl:.2f}, "
        f"Val TokenAcc: {val_token_acc:.3f}, "
        f"Val BLEU: {val_bleu*100:.2f}"
        + (f", Val chrF2: {val_chrf:.2f}" if val_chrf is not None else "")
    )

    # -------- Save per-epoch checkpoint --------
    ckpt = {
        "epoch": epoch + 1,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict(),
        "hparams": {
            "d_model": config.d_model, "n_heads": config.num_heads,
            "num_layers": config.num_layers, "d_ff": config.d_ff, "max_len": MAX_LEN
        },
    }
    epoch_path = os.path.join(config.MODELS_DIR, f"transformer_attention_epoch{epoch+1}.pt")
    torch.save(ckpt, epoch_path)

    # -------- Save best checkpoint --------
    if val_loss_avg < best_val:
        best_val = val_loss_avg
        best_path = os.path.join(getattr(config, "checkpoint_dir", config.MODELS_DIR), "best.pt")
        torch.save(ckpt, best_path)

# Final save (latest state)
final_path = os.path.join(config.MODELS_DIR, "transformer_attention_final.pt")
torch.save(model.state_dict(), final_path)
print(f"Training complete! Saved latest weights to {final_path}")
