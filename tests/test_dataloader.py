import pytest
import sys
import os


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.config import config
from src.data.dataloader import get_data_ready

def test_get_data_ready_returns_dataloaders():
    train_dl, val_dl, tokenizer_src, tokenizer_tgt = get_data_ready(config)
    assert train_dl is not None
    assert val_dl is not None
    assert tokenizer_src.get_vocab_size() > 0
    assert tokenizer_tgt.get_vocab_size() > 0

def test_bilingual_dataset_sample_structure():
    train_dl, _, _, _ = get_data_ready(config)
    sample = next(iter(train_dl))
    assert "encoder_input" in sample
    assert "decoder_input" in sample
    assert "encoder_mask" in sample
    assert "decoder_mask" in sample
    assert "label" in sample
    assert sample["encoder_input"].shape[-1] == config.max_seq_length
