

import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu

from torchmetrics.functional import accuracy
import math
try:
    import sacrebleu
except:
    sacrebleu = None



from config import config
from transformerModel import Transformer
from dataloader import TranslationDataset, collate_fn, en_tok, fr_tok, UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX, MAX_LEN

print("Device is:", config.device)

train_dataset = TranslationDataset("trian")
train_dataloader = DataLoader(
    train_dataset, batch_size=config.trainBatchSize, shuffle=True, collate_fn=collate_fn
)

val_dataset  = TranslationDataset("val")
val_dataloader = DataLoader(
    val_dataset, batch_size=config.valBatchSize, shuffle=False, collate_fn=collate_fn
)
# ================== Train ==================
SRC_VOCAB_SIZE = en_tok.get_vocab_size()
TGT_VOCAB_SIZE = fr_tok.get_vocab_size()
print("Vocab sizes:", SRC_VOCAB_SIZE, TGT_VOCAB_SIZE)

model = Transformer(
    SRC_VOCAB_SIZE, TGT_VOCAB_SIZE,
    d_model=config.d_model, n_heads=config.num_heads, num_layers=config.num_layers, d_ff=config.d_ff, max_len=MAX_LEN
).to(config.device)

optimizer = optim.Adam(model.parameters(), lr=config.lr, betas=(0.9, 0.98), eps=1e-9)
criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

torch.manual_seed(42)
scaler = torch.cuda.amp.GradScaler(enabled=(config.device.type=="cuda"))
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)



for epoch in range(config.num_epochs):
    model.train()
    epoch_loss = 0.0
    progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}")
    for i, (src, tgt) in enumerate(progress_bar):
        src, tgt = src.to(config.device), tgt.to(config.device)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=(config.device.type == "cuda")):
            logits = model(src, tgt[:-1, :])            # [T-1,B,V]
            V = logits.shape[-1]
            loss = criterion(logits.reshape(-1, V), tgt[1:, :].reshape(-1))
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        epoch_loss += float(loss.item())
        progress_bar.set_postfix(batch_loss=float(loss.item()), avg_loss=epoch_loss / (i + 1))
            

    torch.save(model.state_dict(), os.path.join(config.MODELS_DIR, f"transformer_attention_epoch{epoch+1}.pt"))

    # ---- Validation: teacher-forced BLEU on the real val set ----
    # ---- Validation (loss, ppl, token-acc, BLEU, chrF) ----
    model.eval()
    val_references, val_hypotheses = [], []
    val_loss, val_steps = 0.0, 0
    all_preds, all_tgts = [], []

    with torch.no_grad():
        for j, (src, tgt) in enumerate(val_dataloader):
            src, tgt = src.to(config.device), tgt.to(config.device)
            out = model(src, tgt[:-1, :])                # [T-1,B,V]
            V = out.shape[-1]

            # loss
            batch_loss = criterion(out.reshape(-1, V), tgt[1:, :].reshape(-1)).item()
            val_loss += float(batch_loss); val_steps += 1

            # token-level accuracy (teacher-forced)
            pred_ids = out.argmax(-1)                    # [T-1,B]
            mask = (tgt[1:, :] != PAD_IDX)
            all_preds.append(pred_ids[mask])
            all_tgts.append(tgt[1:, :][mask])

            # text for BLEU/chrF
            out_tokens = pred_ids
            for b in range(tgt.size(1)):
                ref = [fr_tok.id_to_token(int(idx))
                    for idx in tgt[1:, b] if int(idx) not in (PAD_IDX, EOS_IDX)]
                hyp = [fr_tok.id_to_token(int(idx))
                    for idx in out_tokens[:, b] if int(idx) not in (PAD_IDX, EOS_IDX)]
                val_references.append([ref])
                val_hypotheses.append(hyp)

    val_loss_avg = (val_loss / max(val_steps, 1))
    val_ppl = math.exp(val_loss_avg) if val_loss_avg < 20 else float("inf")

    if all_preds:
        all_preds = torch.cat(all_preds)
        all_tgts  = torch.cat(all_tgts)
        val_token_acc = float((all_preds == all_tgts).float().mean().item())
    else:
        val_token_acc = 0.0

    val_bleu = corpus_bleu(val_references, val_hypotheses)
    val_chrf = None
    if sacrebleu:
        refs = [" ".join(r[0]) for r in val_references]
        hyps = [" ".join(h)    for h in val_hypotheses]
        val_chrf = sacrebleu.corpus_chrf(hyps, [refs]).score
    
    print(
        f"Epoch {epoch+1}, "
        f"Train Loss: {epoch_loss/len(train_dataloader):.3f}, "
        f"Val Loss: {val_loss_avg:.3f}, "
        f"Val PPL: {val_ppl:.2f}, "
        f"Val TokenAcc: {val_token_acc:.3f}, "
        f"Val BLEU: {val_bleu*100:.2f}"
        + (f", Val chrF2: {val_chrf:.2f}" if val_chrf is not None else "")
    )
    scheduler.step(val_loss_avg)



torch.save(model.state_dict(), os.path.join(config.MODELS_DIR, "transformer_attention_final.pt"))
print("Training complete! Model saved as models/transformer_attention_final.pt")

