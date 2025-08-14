import torch
import os
import sys
from torch.utils.data import DataLoader, Dataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
import pandas as pd
from torch.nn.utils.rnn import pad_sequence

from config import config



def train_wordlevel_tokenizer(sentences, save_path, vocab_size=32000):
    SPECIALS = ["<unk>", "<pad>", "<bos>", "<eos>"]  # tokenizer special tokens (order matters)
    tok = Tokenizer(models.WordLevel(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.WordLevelTrainer(
        vocab_size=vocab_size,
        min_frequency=1,
        special_tokens = SPECIALS
    )
    tok.train_from_iterator((str(s) for s in sentences), trainer=trainer)   

    tok.save(save_path)
    print(f"[Tokenizer] saved at: {save_path}")

def seq_len_with_specials(text, tok):
    return 2 + len(tok.encode(str(text)).ids)  # BOS + EOS

def add_bos_eos(ids):
    return [BOS_IDX] + ids + [EOS_IDX]

class TranslationDataset(Dataset):
    def __init__(self, datagroup):
        if datagroup == "train":
            self.data = df_train.reset_index(drop=True)
        elif datagroup == "test":
            self.data = df_test.reset_index(drop=True)
        else: 
            self.data = df_val.reset_index(drop=True)

        self.en_tok = en_tok
        self.fr_tok = fr_tok
        
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        src_text = str(self.data.iloc[idx][config.sourceLang])
        tgt_text = str(self.data.iloc[idx][config.targetLang])

        max_inner_len = max(2, config.max_seq_length - 2)  # room for BOS/EOS
        src_ids = self.en_tok.encode(src_text).ids[:max_inner_len]
        tgt_ids = self.fr_tok.encode(tgt_text).ids[:max_inner_len]
        src_tensor = torch.tensor([SRC_BOS_IDX] + src_ids + [SRC_EOS_IDX], dtype=torch.long)
        tgt_tensor = torch.tensor([TGT_BOS_IDX] + tgt_ids + [TGT_EOS_IDX], dtype=torch.long)

        return src_tensor, tgt_tensor

def collate_fn(batch):
    # drop degenerate pairs
    batch = [(s, t) for (s, t) in batch if s.numel() > 0 and t.numel() > 0]
    if not batch:
        batch = [
            (torch.tensor([SRC_BOS_IDX, SRC_EOS_IDX], dtype=torch.long),
             torch.tensor([TGT_BOS_IDX, TGT_EOS_IDX], dtype=torch.long))
        ]

    src_batch, tgt_batch = zip(*batch)
    # pad_sequence default: batch_first=False -> [T, B]
    src_batch = pad_sequence(src_batch, padding_value=SRC_PAD_IDX)
    tgt_batch = pad_sequence(tgt_batch, padding_value=TGT_PAD_IDX)
    return src_batch, tgt_batch


# ================== Load Data ==================
csv_train_path = os.path.join(config.DATA_DIR, "opus100_en_fr_train.csv")
df_train = pd.read_csv(csv_train_path)
# df_train = df_train.head(5000)

csv_test_path = os.path.join(config.DATA_DIR, "opus100_en_fr_test.csv")
df_test = pd.read_csv(csv_test_path)

csv_val_path = os.path.join(config.DATA_DIR, "opus100_en_fr_val.csv")
df_val = pd.read_csv(csv_val_path)

# ================== Train & Save Tokenizers (from your data) ==================
en_tokenizer_path = os.path.join(config.TOK_DIR, "tokenizer_en.json")
fr_tokenizer_path = os.path.join(config.TOK_DIR, "tokenizer_fr.json")

if not (os.path.exists(en_tokenizer_path) and os.path.exists(fr_tokenizer_path)):
    en_sentences = df_train[config.sourceLang].dropna().astype(str).tolist()
    fr_sentences = df_train[config.targetLang].dropna().astype(str).tolist()
    train_wordlevel_tokenizer(en_sentences, en_tokenizer_path)
    train_wordlevel_tokenizer(fr_sentences, fr_tokenizer_path)

# ================== Load Tokenizers & Special IDs ==================
en_tok = Tokenizer.from_file(en_tokenizer_path)
fr_tok = Tokenizer.from_file(fr_tokenizer_path)

max_inner_len = max(2, config.max_seq_length - 2)
en_tok.enable_truncation(max_inner_len)
fr_tok.enable_truncation(max_inner_len)

SRC_UNK_IDX = en_tok.token_to_id("<unk>")
SRC_PAD_IDX = en_tok.token_to_id("<pad>")
SRC_BOS_IDX = en_tok.token_to_id("<bos>")
SRC_EOS_IDX = en_tok.token_to_id("<eos>")
# Target (FR)
TGT_UNK_IDX = fr_tok.token_to_id("<unk>")
TGT_PAD_IDX = fr_tok.token_to_id("<pad>")
TGT_BOS_IDX = fr_tok.token_to_id("<bos>")
TGT_EOS_IDX = fr_tok.token_to_id("<eos>")
assert None not in (SRC_UNK_IDX, SRC_PAD_IDX, SRC_BOS_IDX, SRC_EOS_IDX, TGT_UNK_IDX, TGT_PAD_IDX, TGT_BOS_IDX, TGT_EOS_IDX), "Special tokens missing."

MAX_LEN = int(config.max_seq_length)
print("Max sequence length (with BOS/EOS):", MAX_LEN)
