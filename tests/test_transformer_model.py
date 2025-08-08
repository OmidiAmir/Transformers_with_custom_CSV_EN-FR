import torch
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.models.transformer_model import Transformer

def test_transformer_forward_pass():
    # Dummy config
    class DummyConfig:
        d_model = 128
        num_heads = 8
        num_layers = 2
        d_ff = 256
        dropout = 0.1
        max_seq_length = 20
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    config = DummyConfig()

    src_vocab_size = 100
    tgt_vocab_size = 120
    batch_size = 4
    seq_len = 20

    model = Transformer(src_vocab_size, tgt_vocab_size, config)
    model.eval()

    src = torch.randint(1, src_vocab_size, (batch_size, seq_len))
    tgt = torch.randint(1, tgt_vocab_size, (batch_size, seq_len))

    output, enc_output, dec_output = model(src, tgt)

    assert output.shape == (batch_size, seq_len, tgt_vocab_size), f"Output shape incorrect: {output.shape}"
    assert enc_output.shape == (batch_size, seq_len, config.d_model)
    assert dec_output.shape == (batch_size, seq_len, config.d_model)
