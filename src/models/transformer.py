import torch
import torch.nn as nn
import math

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embed_size, num_heads, num_layers,
                 max_length=1024, dropout=0.1, pad_idx=1):
        super(TransformerModel, self).__init__()
        self.embed_size = embed_size
        self.pad_idx = pad_idx

        # Token embedding with padding_idx
        self.token_embedding = nn.Embedding(vocab_size, embed_size, padding_idx=pad_idx)

        # Positional encoding (fixed, not learned)
        self.positional_encoding = nn.Parameter(
            self._generate_positional_encoding(max_length, embed_size),
            requires_grad=False
        )

        # Transformer encoder with batch_first=True
        self.transformer = nn.Transformer(
            d_model=embed_size,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )

        self.dropout = nn.Dropout(dropout)

    def _generate_positional_encoding(self, max_length, embed_size):
        position = torch.arange(0, max_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_size, 2) * -(math.log(10000.0) / embed_size))
        pe = torch.zeros(max_length, embed_size)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # shape: (1, max_length, embed_size)

    def forward(self, src):
        """
        src: Tensor of shape (batch_size, seq_len) with token IDs
        returns: Tensor of shape (batch_size, embed_size) after padding-aware mean pooling
        """
        # 1. Generate padding mask
        src_key_padding_mask = (src == self.pad_idx)

        # 2. Embedding and positional encoding
        src_emb = self.token_embedding(src) * math.sqrt(self.embed_size)
        src_emb = src_emb + self.positional_encoding[:, :src.size(1), :]
        src_emb = self.dropout(src_emb)  # (batch_size, seq_len, embed_dim)

        # 3. Transformer
        output = self.transformer(
            src=src_emb,
            tgt=src_emb,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=src_key_padding_mask
        )  # (batch_size, seq_len, embed_size)

        # 4. Padding-aware mean pooling
        mask = (~src_key_padding_mask).unsqueeze(-1).float()  # (batch_size, seq_len, 1)
        masked_output = output * mask
        summed = masked_output.sum(dim=1)  # (batch_size, embed_size)
        lengths = mask.sum(dim=1).clamp(min=1)  # (batch_size, 1)
        pooled = summed / lengths  # (batch_size, embed_size)

        return pooled