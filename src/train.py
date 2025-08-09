import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import spacy
import nltk
from nltk.translate.bleu_score import corpus_bleu

# NEW: tokenizers (for saving EN/FR tokenizers)
from tokenizers import Tokenizer, models, trainers, pre_tokenizers

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device is: ", DEVICE)

nltk.download('punkt', quiet=True)

# === Language and Special Tokens ===
SRC_LANGUAGE = 'en'
TGT_LANGUAGE = 'fr'
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']

# === Resolve project root & paths (works from root/ or src/)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# === Load Data ===
csv_path = os.path.join(DATA_DIR, "opus100_en_fr_train.csv")
df = pd.read_csv(csv_path)

# === Train & Save WordLevel tokenizers (EN/FR) into project_root/data ===
en_tokenizer_path = os.path.join(DATA_DIR, "tokenizer_en.json")
fr_tokenizer_path = os.path.join(DATA_DIR, "tokenizer_fr.json")

def train_wordlevel_tokenizer(sentences, save_path, vocab_size=32000):
    tok = Tokenizer(models.WordLevel(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.WordLevelTrainer(
        vocab_size=vocab_size,
        min_frequency=1,
        special_tokens=["<unk>", "<pad>", "<bos>", "<eos>"]  # IDs 0..3 to match constants
    )
    tok.train_from_iterator((str(s) for s in sentences), trainer=trainer)
    tok.save(save_path)
    print(f"Tokenizer saved at: {save_path}")

# Collect sentences and train/save tokenizers
en_sentences = df[SRC_LANGUAGE].dropna().astype(str).tolist()
fr_sentences = df[TGT_LANGUAGE].dropna().astype(str).tolist()
train_wordlevel_tokenizer(en_sentences, en_tokenizer_path)
train_wordlevel_tokenizer(fr_sentences, fr_tokenizer_path)

# === Only use a subset for quick debugging/trial ===
N = 10000  # Change this to use a different number of samples
df_sample = df.head(N)  # or use df.sample(N) for a random subset

print(f"Training on {len(df_sample)} samples (subset of dataset).")

# === Tokenizers (spaCy) for current pipeline ===
spacy_en = spacy.load('en_core_web_sm')
spacy_fr = spacy.load('fr_core_news_sm')

def tokenize_en(text):
    return [tok.text.lower() for tok in spacy_en.tokenizer(str(text))]

def tokenize_fr(text):
    return [tok.text.lower() for tok in spacy_fr.tokenizer(str(text))]

token_transform = {SRC_LANGUAGE: tokenize_en, TGT_LANGUAGE: tokenize_fr}

# === Check max sequence lengths ===
max_en_len = df_sample['en'].apply(lambda x: len(token_transform['en'](str(x)))).max()
max_fr_len = df_sample['fr'].apply(lambda x: len(token_transform['fr'](str(x)))).max()
print("Max English tokens:", max_en_len)
print("Max French tokens:", max_fr_len)
# We'll use the max of these, add 10 for safety
max_len = max(max_en_len, max_fr_len) + 10

# === Vocab Building (torchtext) ===
from torchtext.vocab import build_vocab_from_iterator

def yield_tokens(df_frame, language):
    for text in df_frame[language]:
        yield token_transform[language](str(text))

vocab_transform = {}
for lang in [SRC_LANGUAGE, TGT_LANGUAGE]:
    vocab_transform[lang] = build_vocab_from_iterator(
        yield_tokens(df_sample, lang),
        min_freq=2,
        specials=special_symbols,
        special_first=True
    )
    vocab_transform[lang].set_default_index(UNK_IDX)

# === SAVE VOCAB for inference ===
torch.save(vocab_transform, os.path.join(DATA_DIR, "vocab_transform.pt"))
print("Vocabulary saved as data/vocab_transform.pt!")

# === Text to Tensor (for Dataset) ===
def tensor_transform(token_ids):
    return torch.cat((
        torch.tensor([BOS_IDX]),
        torch.tensor(token_ids),
        torch.tensor([EOS_IDX])
    ))

# === Custom Dataset ===
class TranslationDataset(Dataset):
    def __init__(self, dataframe):
        self.data = dataframe.reset_index(drop=True)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        src_text = str(self.data.iloc[idx]['en'])
        tgt_text = str(self.data.iloc[idx]['fr'])
        src_tensor = tensor_transform(vocab_transform[SRC_LANGUAGE](token_transform[SRC_LANGUAGE](src_text)))
        tgt_tensor = tensor_transform(vocab_transform[TGT_LANGUAGE](token_transform[TGT_LANGUAGE](tgt_text)))
        return src_tensor, tgt_tensor

train_dataset = TranslationDataset(df_sample)

# === Collate Function for Dataloader ===
def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)
    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX)
    tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX)
    return src_batch, tgt_batch

BATCH_SIZE = 32
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

# ========== Transformer Components ==========
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
    def forward(self, q, k, v, mask=None):
        batch_size, q_len, d_model = q.size()
        k_len = k.size(1)
        v_len = v.size(1)
        Q = self.q_proj(q).view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(k).view(batch_size, k_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(v).view(batch_size, v_len, self.num_heads, self.head_dim).transpose(1, 2)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_out = torch.matmul(attn_weights, V)
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, q_len, d_model)
        return self.out_proj(attn_out)

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, src_mask=None):
        x2 = self.self_attn(x, x, x, src_mask)
        x = self.norm1(x + self.dropout(x2))
        x2 = self.ff(x)
        x = self.norm2(x + self.dropout(x2))
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.enc_attn = MultiHeadAttention(d_model, num_heads)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, enc_out, tgt_mask=None, memory_mask=None):
        x2 = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(x2))
        x2 = self.enc_attn(x, enc_out, enc_out, memory_mask)
        x = self.norm2(x + self.dropout(x2))
        x2 = self.ff(x)
        x = self.norm3(x + self.dropout(x2))
        return x

def generate_subsequent_mask(size):
    mask = torch.tril(torch.ones(size, size)).unsqueeze(0)
    return mask

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=256, n_heads=4, num_layers=2, d_ff=512, dropout=0.1, max_len=512):
        super().__init__()
        self.src_embedding = nn.Embedding(src_vocab_size, d_model, padding_idx=PAD_IDX)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model, padding_idx=PAD_IDX)
        self.pos_encoder = PositionalEncoding(d_model, max_len)
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(num_layers)])
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
    def encode(self, src, src_mask=None):
        x = self.src_embedding(src)
        x = self.pos_encoder(x)
        for layer in self.encoder_layers:
            x = layer(x, src_mask)
        return x
    def decode(self, tgt, memory, tgt_mask=None, memory_mask=None):
        x = self.tgt_embedding(tgt)
        x = self.pos_encoder(x)
        for layer in self.decoder_layers:
            x = layer(x, memory, tgt_mask, memory_mask)
        return self.fc_out(x)
    def forward(self, src, tgt):
        src = src.transpose(0, 1)
        tgt = tgt.transpose(0, 1)
        memory = self.encode(src)
        seq_len = tgt.size(1)
        tgt_mask = generate_subsequent_mask(seq_len).to(tgt.device)
        out = self.decode(tgt, memory, tgt_mask=tgt_mask)
        return out.transpose(0, 1)

# ========== Training ==========
SRC_VOCAB_SIZE = len(vocab_transform[SRC_LANGUAGE])
TGT_VOCAB_SIZE = len(vocab_transform[TGT_LANGUAGE])
D_MODEL = 256
N_HEADS = 4
N_LAYERS = 2
D_FF = 512

model = Transformer(SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, d_model=D_MODEL, n_heads=N_HEADS, num_layers=N_LAYERS, d_ff=D_FF, max_len=max_len).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

EPOCHS = 10
for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0
    progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}")
    for i, (src, tgt) in enumerate(progress_bar):
        src, tgt = src.to(DEVICE), tgt.to(DEVICE)
        optimizer.zero_grad()
        output = model(src, tgt[:-1, :])
        output_dim = output.shape[-1]
        output = output.reshape(-1, output_dim)
        tgt_y = tgt[1:, :].reshape(-1)
        loss = criterion(output, tgt_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        progress_bar.set_postfix(batch_loss=loss.item(), avg_loss=epoch_loss/(i+1))
    torch.save(model.state_dict(), os.path.join(MODELS_DIR, f"transformer_attention_epoch{epoch+1}.pt"))

    # BLEU evaluation (quick, small sample; still teacher-forced)
    model.eval()
    references, hypotheses = [], []
    with torch.no_grad():
        for src, tgt in list(train_dataloader)[:5]:
            src, tgt = src.to(DEVICE), tgt.to(DEVICE)
            output = model(src, tgt[:-1, :])
            out_tokens = output.argmax(-1)
            for i_seq in range(tgt.size(1)):
                ref = [vocab_transform[TGT_LANGUAGE].lookup_token(idx.item()) for idx in tgt[1:, i_seq] if idx.item() != PAD_IDX and idx.item() != EOS_IDX]
                hyp = [vocab_transform[TGT_LANGUAGE].lookup_token(idx.item()) for idx in out_tokens[:, i_seq] if idx.item() != PAD_IDX and idx.item() != EOS_IDX]
                references.append([ref])
                hypotheses.append(hyp)
    bleu = corpus_bleu(references, hypotheses)
    print(f"Epoch {epoch+1}, Avg Loss: {epoch_loss/len(train_dataloader):.3f}, TF BLEU-ish: {bleu*100:.2f}")

torch.save(model.state_dict(), os.path.join(MODELS_DIR, "transformer_attention_final.pt"))
print("Training complete! Model saved as models/transformer_attention_final.pt")














# import os
# import math
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# import pandas as pd
# from torch.utils.data import DataLoader, Dataset
# from torch.nn.utils.rnn import pad_sequence
# from tqdm import tqdm
# import spacy
# import nltk
# from nltk.translate.bleu_score import corpus_bleu
# from torchtext.vocab import build_vocab_from_iterator  # 0.18 API

# # ========= Setup =========
# DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print("Device is:", DEVICE)
# nltk.download('punkt', quiet=True)

# # === Language + Special Tokens ===
# SRC_LANGUAGE = 'en'
# TGT_LANGUAGE = 'fr'
# UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
# SPECIALS = ['<unk>', '<pad>', '<bos>', '<eos>']

# # === Tokenizers (spaCy) ===
# spacy_en = spacy.load('en_core_web_sm')
# spacy_fr = spacy.load('fr_core_news_sm')

# def tokenize_en(text):
#     return [tok.text.lower() for tok in spacy_en.tokenizer(str(text))]

# def tokenize_fr(text):
#     return [tok.text.lower() for tok in spacy_fr.tokenizer(str(text))]

# token_transform = {SRC_LANGUAGE: tokenize_en, TGT_LANGUAGE: tokenize_fr}

# # ========= Data =========
# csv_path = "data/opus100_en_fr_train.csv"
# df = pd.read_csv(csv_path)

# # Use a subset for quick trials
# N = 10000
# df_sample = df.head(N)
# print(f"Training on {len(df_sample)} samples (subset).")

# # Max lengths (quick stats)
# max_en_len = df_sample['en'].apply(lambda x: len(token_transform['en'](str(x)))).max()
# max_fr_len = df_sample['fr'].apply(lambda x: len(token_transform['fr'](str(x)))).max()
# print("Max English tokens:", max_en_len)
# print("Max French tokens:", max_fr_len)
# max_len = int(max(max_en_len, max_fr_len) + 10)

# # ========= Vocab (0.18 style) =========
# def yield_tokens(df, language):
#     for text in df[language]:
#         yield token_transform[language](str(text))

# vocab_transform = {}
# for lang in [SRC_LANGUAGE, TGT_LANGUAGE]:
#     v = build_vocab_from_iterator(
#         yield_tokens(df_sample, lang),
#         min_freq=2,
#         specials=SPECIALS,
#         special_first=True
#     )
#     # IMPORTANT in 0.18: set default index to UNK token *id*, not a hardcoded int
#     v.set_default_index(v['<unk>'])
#     vocab_transform[lang] = v

# # Save vocab
# os.makedirs("data", exist_ok=True)
# torch.save(vocab_transform, "data/vocab_transform.pt")
# print("Vocabulary saved as data/vocab_transform.pt")

# # ========= Dataset / Collate =========
# def tensor_transform(token_ids):
#     return torch.cat((
#         torch.tensor([BOS_IDX], dtype=torch.long),
#         torch.tensor(token_ids, dtype=torch.long),
#         torch.tensor([EOS_IDX], dtype=torch.long)
#     ))

# class TranslationDataset(Dataset):
#     def __init__(self, dataframe):
#         self.data = dataframe.reset_index(drop=True)
#     def __len__(self):
#         return len(self.data)
#     def __getitem__(self, idx):
#         src_text = str(self.data.iloc[idx]['en'])
#         tgt_text = str(self.data.iloc[idx]['fr'])
#         src_ids = [vocab_transform[SRC_LANGUAGE][t] for t in token_transform[SRC_LANGUAGE](src_text)]
#         tgt_ids = [vocab_transform[TGT_LANGUAGE][t] for t in token_transform[TGT_LANGUAGE](tgt_text)]
#         # Drop truly empty sequences (rare but prevents NaNs)
#         if len(src_ids) == 0: src_ids = []
#         if len(tgt_ids) == 0: tgt_ids = []
#         src_tensor = tensor_transform(src_ids)
#         tgt_tensor = tensor_transform(tgt_ids)
#         return src_tensor, tgt_tensor

# train_dataset = TranslationDataset(df_sample)

# def collate_fn(batch):
#     # Filter out any degenerate pairs
#     batch = [(s, t) for (s, t) in batch if len(s) > 0 and len(t) > 0]
#     if not batch:
#         batch = [(
#             torch.tensor([BOS_IDX, EOS_IDX], dtype=torch.long),
#             torch.tensor([BOS_IDX, EOS_IDX], dtype=torch.long)
#         )]
#     src_batch, tgt_batch = zip(*batch)
#     # pad_sequence default: batch_first=False -> [T, B]
#     src_batch = pad_sequence(src_batch, padding_value=PAD_IDX)
#     tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX)
#     return src_batch, tgt_batch

# BATCH_SIZE = 32
# train_dataloader = DataLoader(
#     train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn
# )

# # ========= Transformer =========
# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model: int, max_len: int):
#         super().__init__()
#         import math as _math
#         max_len = int(max_len)
#         pe = torch.zeros(max_len, d_model, dtype=torch.float32)
#         position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
#         div_term = torch.exp(
#             torch.arange(0, d_model, 2, dtype=torch.float32) * (-_math.log(10000.0) / d_model)
#         )
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         self.register_buffer('pe', pe.unsqueeze(0), persistent=False)
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         # x: [B, T, D] here (we arrange below accordingly)
#         return x + self.pe[:, :x.size(1)]

# class MultiHeadAttention(nn.Module):
#     def __init__(self, d_model, num_heads):
#         super().__init__()
#         assert d_model % num_heads == 0
#         self.d_model = d_model
#         self.num_heads = num_heads
#         self.head_dim = d_model // num_heads
#         self.q_proj = nn.Linear(d_model, d_model)
#         self.k_proj = nn.Linear(d_model, d_model)
#         self.v_proj = nn.Linear(d_model, d_model)
#         self.out_proj = nn.Linear(d_model, d_model)
#     def forward(self, q, k, v, mask=None):
#         # q,k,v: [B, T, D]
#         B, Tq, D = q.size()
#         Tk = k.size(1)
#         Tv = v.size(1)
#         Q = self.q_proj(q).view(B, Tq, self.num_heads, self.head_dim).transpose(1, 2)  # [B,H,Tq,hd]
#         K = self.k_proj(k).view(B, Tk, self.num_heads, self.head_dim).transpose(1, 2)   # [B,H,Tk,hd]
#         V = self.v_proj(v).view(B, Tv, self.num_heads, self.head_dim).transpose(1, 2)   # [B,H,Tv,hd]
#         scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)        # [B,H,Tq,Tk]
#         if mask is not None:
#             # mask: bool, True=keep, False=mask
#             neg_inf = torch.finfo(scores.dtype).min
#             scores = scores.masked_fill(~mask, neg_inf)
#         attn = F.softmax(scores, dim=-1)
#         out = torch.matmul(attn, V)                                                     # [B,H,Tq,hd]
#         out = out.transpose(1, 2).contiguous().view(B, Tq, D)                           # [B,Tq,D]
#         return self.out_proj(out)

# class FeedForward(nn.Module):
#     def __init__(self, d_model, d_ff, dropout=0.1):
#         super().__init__()
#         self.linear1 = nn.Linear(d_model, d_ff)
#         self.dropout = nn.Dropout(dropout)
#         self.linear2 = nn.Linear(d_ff, d_model)
#     def forward(self, x):
#         return self.linear2(self.dropout(F.relu(self.linear1(x))))

# class EncoderLayer(nn.Module):
#     def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
#         super().__init__()
#         self.self_attn = MultiHeadAttention(d_model, num_heads)
#         self.norm1 = nn.LayerNorm(d_model)
#         self.ff = FeedForward(d_model, d_ff, dropout)
#         self.norm2 = nn.LayerNorm(d_model)
#         self.dropout = nn.Dropout(dropout)
#     def forward(self, x, src_mask=None):
#         x2 = self.self_attn(x, x, x, src_mask)
#         x = self.norm1(x + self.dropout(x2))
#         x2 = self.ff(x)
#         x = self.norm2(x + self.dropout(x2))
#         return x

# class DecoderLayer(nn.Module):
#     def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
#         super().__init__()
#         self.self_attn = MultiHeadAttention(d_model, num_heads)
#         self.norm1 = nn.LayerNorm(d_model)
#         self.enc_attn = MultiHeadAttention(d_model, num_heads)
#         self.norm2 = nn.LayerNorm(d_model)
#         self.ff = FeedForward(d_model, d_ff, dropout)
#         self.norm3 = nn.LayerNorm(d_model)
#         self.dropout = nn.Dropout(dropout)
#     def forward(self, x, enc_out, tgt_mask=None, memory_mask=None):
#         x2 = self.self_attn(x, x, x, tgt_mask)
#         x = self.norm1(x + self.dropout(x2))
#         x2 = self.enc_attn(x, enc_out, enc_out, memory_mask)
#         x = self.norm2(x + self.dropout(x2))
#         x2 = self.ff(x)
#         x = self.norm3(x + self.dropout(x2))
#         return x

# def generate_subsequent_mask(T: int, device=None):
#     # Bool mask: True = allowed, False = masked
#     m = torch.ones((T, T), dtype=torch.bool, device=device).tril()
#     # Expand to [B=1, H=1, T, T] for broadcasting over batch and heads
#     return m.unsqueeze(0).unsqueeze(0)

# class Transformer(nn.Module):
#     def __init__(self, src_vocab_size, tgt_vocab_size, d_model=256, n_heads=4, num_layers=2, d_ff=512, dropout=0.1, max_len=512):
#         super().__init__()
#         self.src_embedding = nn.Embedding(src_vocab_size, d_model, padding_idx=PAD_IDX)
#         self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model, padding_idx=PAD_IDX)
#         self.pos_encoder = PositionalEncoding(d_model, max_len)
#         self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(num_layers)])
#         self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(num_layers)])
#         self.fc_out = nn.Linear(d_model, tgt_vocab_size)
#     def encode(self, src, src_mask=None):
#         # src: [T, B] -> [B, T, D]
#         x = self.src_embedding(src.transpose(0, 1))
#         x = self.pos_encoder(x)
#         for layer in self.encoder_layers:
#             x = layer(x, src_mask)
#         return x
#     def decode(self, tgt, memory, tgt_mask=None, memory_mask=None):
#         # tgt: [T, B] -> [B, T, D]
#         x = self.tgt_embedding(tgt.transpose(0, 1))
#         x = self.pos_encoder(x)
#         for layer in self.decoder_layers:
#             x = layer(x, memory, tgt_mask, memory_mask)
#         return self.fc_out(x)  # [B, T, V]
#     def forward(self, src, tgt):
#         memory = self.encode(src)
#         Tt = tgt.size(0)
#         tgt_mask = generate_subsequent_mask(Tt, device=tgt.device)
#         out = self.decode(tgt, memory, tgt_mask=tgt_mask)  # [B, T, V]
#         return out.transpose(0, 1)  # -> [T, B, V]

# # ========= Training =========
# SRC_VOCAB_SIZE = len(vocab_transform[SRC_LANGUAGE])
# TGT_VOCAB_SIZE = len(vocab_transform[TGT_LANGUAGE])

# D_MODEL = 256
# N_HEADS = 4
# N_LAYERS = 2
# D_FF = 512

# model = Transformer(
#     SRC_VOCAB_SIZE, TGT_VOCAB_SIZE,
#     d_model=D_MODEL, n_heads=N_HEADS, num_layers=N_LAYERS, d_ff=D_FF, max_len=max_len
# ).to(DEVICE)

# optimizer = optim.Adam(model.parameters(), lr=3e-4, betas=(0.9, 0.98), eps=1e-9)
# criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

# os.makedirs("models", exist_ok=True)

# EPOCHS = 3
# for epoch in range(EPOCHS):
#     model.train()
#     epoch_loss = 0.0
#     progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}")
#     for i, (src, tgt) in enumerate(progress_bar):
#         src, tgt = src.to(DEVICE), tgt.to(DEVICE)
#         optimizer.zero_grad()
#         # Teacher forcing: input = tgt[:-1], predict tgt[1:]
#         logits = model(src, tgt[:-1, :])          # [T-1, B, V]
#         V = logits.shape[-1]
#         loss = criterion(logits.reshape(-1, V), tgt[1:, :].reshape(-1))
#         loss.backward()
#         torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
#         optimizer.step()
#         epoch_loss += float(loss.item())
#         progress_bar.set_postfix(batch_loss=float(loss.item()), avg_loss=epoch_loss / (i + 1))

#     torch.save(model.state_dict(), f"models/transformer_attention_epoch{epoch+1}.pt")

#     # ---- Quick BLEU-ish eval on a few batches ----
#     model.eval()
#     references, hypotheses = [], []
#     with torch.no_grad():
#         for j, (src, tgt) in enumerate(train_dataloader):
#             if j >= 5: break
#             src, tgt = src.to(DEVICE), tgt.to(DEVICE)
#             out = model(src, tgt[:-1, :])            # [T-1, B, V]
#             out_tokens = out.argmax(-1)              # [T-1, B]
#             for b in range(tgt.size(1)):
#                 ref = [vocab_transform[TGT_LANGUAGE].lookup_token(int(idx))
#                        for idx in tgt[1:, b] if int(idx) not in (PAD_IDX, EOS_IDX)]
#                 hyp = [vocab_transform[TGT_LANGUAGE].lookup_token(int(idx))
#                        for idx in out_tokens[:, b] if int(idx) not in (PAD_IDX, EOS_IDX)]
#                 references.append([ref])
#                 hypotheses.append(hyp)
#     bleu = corpus_bleu(references, hypotheses)
#     print(f"Epoch {epoch+1}, Avg Loss: {epoch_loss/len(train_dataloader):.3f}, BLEU: {bleu*100:.2f}")

# torch.save(model.state_dict(), "models/transformer_attention_final.pt")
# print("Training complete! Saved to models/transformer_attention_final.pt")
