import os
from pathlib import Path
import pandas as pd
import torch
import torchmetrics
import sys

from sklearn.model_selection import train_test_split
import pyarrow as pa
from datasets import Dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from torch.utils.data import DataLoader, random_split
from torch.utils.data import Dataset as TDataset


# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# import src.utils.helper 

from src.utils.helper import get_or_build_tokenizer


class BilingualDataset(TDataset):

    def __init__(self, ds, tokenizer_src, tokenizer_tgt, config):
        super().__init__()
        self.seq_len = config.max_seq_length

        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = config.sourceLang
        self.tgt_lang = config.targetLang

        self.sos_token = torch.tensor([tokenizer_tgt.token_to_id("[SOS]")], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer_tgt.token_to_id("[EOS]")], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer_tgt.token_to_id("[PAD]")], dtype=torch.int64)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        src_target_pair = self.ds[idx]
        src_text = src_target_pair[self.src_lang]
        tgt_text = src_target_pair[self.tgt_lang]

        # Transform the text into tokens
        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        # Add sos, eos and padding to each sentence
        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2  # We will add <s> and </s>
        # We will only add <s>, and </s> only on the label
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1

        # Make sure the number of padding tokens is not negative. If it is, the sentence is too long
        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            return self.__getitem__((idx + 1) % len(self.ds))  # skip and move to next


        # Add <s> and </s> token
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        # Add only <s> token
        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        # Add only </s> token
        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        # Double check the size of the tensors to make sure they are all seq_len long
        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        return {
            "encoder_input": encoder_input,  # (seq_len)
            "decoder_input": decoder_input,  # (seq_len)
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), # (1, 1, seq_len)
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).int() & causal_mask(decoder_input.size(0)), # (1, seq_len) & (1, seq_len, seq_len),
            "label": label,  # (seq_len)
            "src_text": src_text,
            "tgt_text": tgt_text,
        }



def causal_mask(size):
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask == 0


def get_data_ready(config):
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    currentPath = os.path.join(project_root, "data")
    path_to_file = os.path.join(currentPath, config.dataSetFileName)
    # project_root=sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..","..")))
    # # currentPath = os.path.dirname(os.path.abspath(__file__))    
    # # path_to_file = Path(f"{currentPath}/{config.dataSetFileName}")
    # currentPath = os.path.join(project_root, "data")
    # path_to_file = os.path.join(currentPath, config.dataSetFileName)
    print("projec root is:>",project_root)
    print("Current path is:>",currentPath)
    print("Path to file is:>",path_to_file)


    csv = pd.read_csv(path_to_file)
    ds_raw = Dataset(pa.Table.from_pandas(csv))
     
    tokenizer_src = get_or_build_tokenizer(ds_raw, config.sourceLang)
    tokenizer_tgt = get_or_build_tokenizer(ds_raw, config.targetLang)

    train_ds_size = int(config.trainPercentage * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

    train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config)
    val_ds   = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, config)

    # Find the maximum length of each sentence in the source and target sentence
    max_len_src = 0
    max_len_tgt = 0

    for item in ds_raw:
        src_ids = tokenizer_src.encode(item['en']).ids
        tgt_ids = tokenizer_tgt.encode(item['fr']).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f'Max length of source sentence: {max_len_src}')
    print(f'Max length of target sentence: {max_len_tgt}')

    train_dataloader = DataLoader(train_ds, batch_size=config.trainBatchSize, shuffle=True)
    val_dataloader   = DataLoader(val_ds, batch_size=config.valBatchSize, shuffle=True)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt
    