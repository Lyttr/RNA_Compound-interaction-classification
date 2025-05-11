import torch
import torch.nn as nn
import math

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embed_size, num_heads, num_layers, max_length=1024, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.embed_size = embed_size
        self.token_embedding = nn.Embedding(vocab_size, embed_size)
        self.positional_encoding = nn.Parameter(self._generate_positional_encoding(max_length, embed_size), requires_grad=False)
        self.transformer = nn.Transformer(d_model=embed_size, nhead=num_heads, num_encoder_layers=num_layers, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def _generate_positional_encoding(self, max_length, embed_size):
        position = torch.arange(0, max_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_size, 2) * -(math.log(10000.0) / embed_size))
        pe = torch.zeros(max_length, embed_size)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def forward(self, src):
        # src shape: (batch_size, seq_length)
        src = self.token_embedding(src) * math.sqrt(self.embed_size)
        src = src + self.positional_encoding[:, :src.size(1), :]
        src = self.dropout(src)
        # Transformer expects input shape: (seq_length, batch_size, embed_size)
        src = src.permute(1, 0, 2)
        output = self.transformer(src, src)
        # Output shape: (seq_length, batch_size, embed_size)
        output = output.permute(1, 0, 2)
        # Apply average pooling over the embedding dimension
        output = output.mean(dim=-1)  # Shape: (batch_size, seq_length)
        return output


