import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataloader import PAD_IDX 
from config import config

# ================== Model ==================
class PositionalEncoding(nn.Module):
    """Auto-extends PE; registers buffer once to avoid KeyError."""
    def __init__(self, d_model, max_len):
        super().__init__()
        self.d_model = d_model
        # register once; fill via _rebuild
        self.register_buffer("pe", torch.empty(1, 0), persistent=False)
        self._rebuild(max_len)

    def _rebuild(self, max_len: int):
        pe = torch.zeros(max_len, self.d_model, dtype=torch.float32)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / self.d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # assign (do NOT register again)
        self.pe = pe.unsqueeze(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        T = x.size(1)
        if T > self.pe.size(1):
            self._rebuild(T)
            self.pe = self.pe.to(x.device)
        return x + self.pe[:, :T]

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
        # q,k,v: [B, T, D]
        B, Tq, D = q.size()
        Tk = k.size(1)
        Q = self.q_proj(q).view(B, Tq, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(k).view(B, Tk, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(v).view(B, Tk, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)        # [B,H,Tq,Tk]
        if mask is not None:
            # mask: bool with True=keep, False=mask
            neg_inf = torch.finfo(scores.dtype).min
            scores = scores.masked_fill(~mask, neg_inf)
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)                                                     # [B,H,Tq,hd]
        out = out.transpose(1, 2).contiguous().view(B, Tq, D)                           # [B,Tq,D]
        return self.out_proj(out)

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

def generate_subsequent_mask(T: int, device=None):
    # Bool mask: True=keep, False=mask
    m = torch.ones((T, T), dtype=torch.bool, device=device).tril()
    return m.unsqueeze(0).unsqueeze(0)  # [1,1,T,T]

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=256, n_heads=4, num_layers=2, d_ff=512, dropout=0.1, max_len=512):
        super().__init__()
        self.src_embedding = nn.Embedding(src_vocab_size, d_model, padding_idx=PAD_IDX)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model, padding_idx=PAD_IDX)
        self.pos_encoder = PositionalEncoding(d_model, max_len)
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, n_heads, d_ff, config.dropout) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, n_heads, d_ff, config.dropout) for _ in range(num_layers)])
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
    def encode(self, src, src_mask=None):
        # src: [T,B] -> [B,T,D]
        x = self.src_embedding(src.transpose(0, 1))
        x = self.pos_encoder(x)
        for layer in self.encoder_layers:
            x = layer(x, src_mask)
        return x
    def decode(self, tgt, memory, tgt_mask=None, memory_mask=None):
        # tgt: [T,B] -> [B,T,D]
        x = self.tgt_embedding(tgt.transpose(0, 1))
        x = self.pos_encoder(x)
        for layer in self.decoder_layers:
            x = layer(x, memory, tgt_mask, memory_mask)
        return self.fc_out(x)  # [B,T,V]
    def forward(self, src, tgt):
        memory = self.encode(src)
        Tt = tgt.size(0)
        tgt_mask = generate_subsequent_mask(Tt, device=tgt.device)
        out = self.decode(tgt, memory, tgt_mask=tgt_mask)  # [B,T,V]
        return out.transpose(0, 1)  # -> [T,B,V]
