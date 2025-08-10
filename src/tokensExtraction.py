import os
from tokenizers import Tokenizer, models, trainers, pre_tokenizers

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