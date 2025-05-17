# lstm.py

import torch
import torch.nn as nn

class LSTMEncoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, bidirectional=False, pad_idx=1):
        super(LSTMEncoder, self).__init__()
        self.pad_idx = pad_idx
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional
        )
        self.output_dim = hidden_size * (2 if bidirectional else 1)

    def forward(self, tokens, mask=None):
        """
        Args:
            tokens (Tensor): (batch_size, seq_len)
            mask (Tensor, optional): (batch_size, seq_len), where 1 is real token, 0 is pad

        Returns:
            Tensor: (batch_size, output_dim) — mean pooled hidden states
        """
        # 1. Auto-generate mask if not provided
        if mask is None:
            mask = (tokens != self.pad_idx).float()  # (batch_size, seq_len)

        # 2. Embedding
        embedded = self.embedding(tokens)  # (batch_size, seq_len, embed_size)

        # 3. LSTM
        outputs, _ = self.lstm(embedded)  # (batch_size, seq_len, hidden_dim)

        # 4. Padding-aware mean pooling
        mask = mask.unsqueeze(-1)  # (batch_size, seq_len, 1)
        outputs = outputs * mask  # zero out pad embeddings
        summed = outputs.sum(dim=1)  # (batch_size, hidden_dim)
        lengths = mask.sum(dim=1).clamp(min=1e-8)  # (batch_size, 1)
        pooled = summed / lengths  # (batch_size, hidden_dim)

        return pooled