import torch
import torch.nn as nn
from src.models.transformer import TransformerModel
from src.models.gnn import GNN_graphpred
from src.models.lstm import LSTMEncoder  
from src.models.image_mol import ImageMol 
import numpy as np
import fm
from pathlib import Path
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=1024):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(0.1)

        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(0.1)

        self.fc3 = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        #x = self.drop1(x)

        x = self.fc2(x)
        x = self.relu2(x)
        #x = self.drop2(x)

        x = self.fc3(x)
        x = self.sigmoid(x)
        return x.squeeze(1)

class TransformerGNN(nn.Module):
    def __init__(self, vocab_size, embed_size, num_heads, num_layers, gnn_config, mlp_hidden_dim):
        super(TransformerGNN, self).__init__()
        self.transformer = TransformerModel(vocab_size, embed_size, num_heads, num_layers)
        self.gnn = GNN_graphpred(**gnn_config)
        self.mlp = MLP(input_dim=embed_size + gnn_config['emb_dim'], hidden_dim=mlp_hidden_dim)

    def forward(self, tokens, graph_data):
        token_embeddings = self.transformer(tokens)  # (batch_size, embed_dim)
        graph_embeddings = self.gnn(graph_data)      # (batch_size, gnn_emb_dim)
        combined = torch.cat((token_embeddings, graph_embeddings), dim=1)
        return self.mlp(combined)

class LSTM_GNN(nn.Module):
    def __init__(self, vocab_size, embed_size, lstm_hidden_size, lstm_layers, gnn_config, mlp_hidden_dim, bidirectional=False):
        super(LSTM_GNN, self).__init__()
        self.lstm = LSTMEncoder(vocab_size, embed_size, lstm_hidden_size, lstm_layers, bidirectional=bidirectional)
        self.gnn = GNN_graphpred(**gnn_config)

        lstm_output_dim = self.lstm.output_dim  
        self.mlp = MLP(input_dim=lstm_output_dim + gnn_config['emb_dim'], hidden_dim=mlp_hidden_dim)

    def forward(self, tokens, graph_data, mask=None):
        token_embeddings = self.lstm(tokens, mask=mask)     # (batch_size, lstm_output_dim)
        graph_embeddings = self.gnn(graph_data)             # (batch_size, gnn_emb_dim)
        combined = torch.cat((token_embeddings, graph_embeddings), dim=1)
        return self.mlp(combined)
class RNAFM_Drugchat(nn.Module):
    def __init__(self, gnn_config,mlp_hidden_dim):
        super(RNAFM_Drugchat, self).__init__()
        data_dir = '../input/rnafm-tutorial/'
        temp_model, alphabet = fm.pretrained.rna_fm_t12(Path(data_dir, 'RNA-FM_pretrained.pth'))
    
        self.fm_model=fm.BioBertModel(temp_model.args,alphabet)
        self.gnn = GNN_graphpred(**gnn_config)

        self.cnn=ImageMol("ResNet18")
        self.mlp = MLP(input_dim=1452, hidden_dim=mlp_hidden_dim)

    def forward(self, tokens, graph_data,image_data):
        token_embeddings = self.fm_model(tokens,repr_layers=[12])['representations'][12]
        token_embeddings = torch.max(token_embeddings, dim=1).values 
        graph_embeddings = self.gnn(graph_data)    
        image_embeddings = self.cnn(image_data)
        combined = torch.cat((token_embeddings, graph_embeddings,image_embeddings), dim=1)
        return self.mlp(combined)