import torch
import torch.nn as nn
from transformer import TransformerModel
from gnn import GNN_graphpred

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
        # Process tokens through the transformer
        token_embeddings = self.transformer(tokens)  # Shape: (batch_size, seq_length)

        # Process graph data through the GNN
        graph_embeddings = self.gnn(graph_data)  # Shape: (batch_size, gnn_emb_dim)

        # Concatenate the embeddings
        combined_embeddings = torch.cat((token_embeddings, graph_embeddings), dim=1)

        # Feed into MLP for classification
        output = self.mlp(combined_embeddings)
        return output

# Example usage
# vocab_size = 10000
# embed_size = 512
# num_heads = 8
# num_layers = 6
# gnn_config = {
#     "num_layer": 5,
#     "emb_dim": 300,
#     "num_tasks": 1,
#     "JK": "last",
#     "graph_pooling": "mean",
#     "gnn_type": "gin"
# }
# mlp_hidden_dim = 1024
# model = TransformerGNN(vocab_size, embed_size, num_heads, num_layers, gnn_config, mlp_hidden_dim)
# tokens = torch.randint(0, vocab_size, (32, 1024))
# graph_data = ...  # Your graph data here
# output = model(tokens, graph_data)
