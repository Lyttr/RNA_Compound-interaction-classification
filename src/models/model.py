import torch
import torch.nn as nn
from src.models.transformer import TransformerModel
from src.models.gnn import GNN_graphpred
from src.models.lstm import LSTMEncoder  
from src.models.image_mol import ImageMol 
import numpy as np
import fm
from pathlib import Path
def print_gpu_memory(prefix=""):
    """打印当前GPU显存使用情况"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3    # GB
        max_allocated = torch.cuda.max_memory_allocated() / 1024**3  # GB
        print(f"{prefix}GPU Memory: Allocated={allocated:.2f}GB, Reserved={reserved:.2f}GB, Max={max_allocated:.2f}GB")
        return allocated, reserved, max_allocated
    return 0, 0, 0
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


        x = self.fc2(x)
        x = self.relu2(x)


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
        token_embeddings = self.transformer(tokens) 
        graph_embeddings = self.gnn(graph_data)     
        combined = torch.cat((token_embeddings, graph_embeddings), dim=1)
        return self.mlp(combined)
class Transformer(nn.Module):
    def __init__(self, vocab_size, embed_size, num_heads, num_layers, mlp_hidden_dim):
        super(Transformer, self).__init__()
        self.transformer = TransformerModel(vocab_size, embed_size, num_heads, num_layers)
      
        self.mlp = MLP(input_dim=embed_size , hidden_dim=mlp_hidden_dim)

    def forward(self, tokens):
        token_embeddings = self.transformer(tokens)     
        return self.mlp(token_embeddings)
class TransformerCNN(nn.Module):
    def __init__(self, vocab_size, embed_size, num_heads, num_layers, mlp_hidden_dim):
        super(TransformerCNN, self).__init__()
        self.transformer = TransformerModel(vocab_size, embed_size, num_heads, num_layers)
        self.cnn=ImageMol("ResNet18")
        self.mlp = MLP(input_dim=embed_size + 512, hidden_dim=mlp_hidden_dim)

    def forward(self, tokens, image_data):
        token_embeddings = self.transformer(tokens) 
        image_embeddings = self.cnn(image_data)   
        combined = torch.cat((token_embeddings, image_embeddings), dim=1)
        return self.mlp(combined)
class LSTM_GNN(nn.Module):
    def __init__(self, vocab_size, embed_size, lstm_hidden_size, lstm_layers, gnn_config, mlp_hidden_dim, bidirectional=False):
        super(LSTM_GNN, self).__init__()
        self.lstm = LSTMEncoder(vocab_size, embed_size, lstm_hidden_size, lstm_layers, bidirectional=bidirectional)
        self.gnn = GNN_graphpred(**gnn_config)

        lstm_output_dim = self.lstm.output_dim  
        self.mlp = MLP(input_dim=lstm_output_dim + gnn_config['emb_dim'], hidden_dim=mlp_hidden_dim)

    def forward(self, tokens, graph_data, mask=None):
        token_embeddings = self.lstm(tokens, mask=mask)     
        graph_embeddings = self.gnn(graph_data)            
        combined = torch.cat((token_embeddings, graph_embeddings), dim=1)
        return self.mlp(combined)
class LSTM(nn.Module):
    def __init__(self, vocab_size, embed_size, lstm_hidden_size, lstm_layers, mlp_hidden_dim, bidirectional=False):
        super(LSTM, self).__init__()
        self.lstm = LSTMEncoder(vocab_size, embed_size, lstm_hidden_size, lstm_layers, bidirectional=bidirectional)
    

        lstm_output_dim = self.lstm.output_dim  
        self.mlp = MLP(input_dim=lstm_output_dim , hidden_dim=mlp_hidden_dim)

    def forward(self, tokens, mask=None):
        token_embeddings = self.lstm(tokens, mask=mask)     
        return self.mlp(token_embeddings)
class LSTM_CNN(nn.Module):
    def __init__(self, vocab_size, embed_size, lstm_hidden_size, lstm_layers, mlp_hidden_dim, bidirectional=False):
        super(LSTM_CNN, self).__init__()
        self.lstm = LSTMEncoder(vocab_size, embed_size, lstm_hidden_size, lstm_layers, bidirectional=bidirectional)
        self.cnn=ImageMol("ResNet18")

        lstm_output_dim = self.lstm.output_dim  
        self.mlp = MLP(input_dim=lstm_output_dim + 512, hidden_dim=mlp_hidden_dim)

    def forward(self, tokens, image_data, mask=None):
        token_embeddings = self.lstm(tokens, mask=mask)     
        image_embeddings = self.cnn(image_data)       
        combined = torch.cat((token_embeddings, image_embeddings), dim=1)
        return self.mlp(combined)
class RNAFM_GNN(nn.Module):
    def __init__(self, gnn_config,mlp_hidden_dim):
        super(RNAFM_GNN, self).__init__()
        data_dir = '../input/rnafm-tutorial/'
        temp_model, alphabet = fm.pretrained.rna_fm_t12(Path(data_dir, 'RNA-FM_pretrained.pth'))
    
        self.fm_model=fm.BioBertModel(temp_model.args,alphabet)
        self.gnn = GNN_graphpred(**gnn_config)

        
        self.mlp = MLP(input_dim=1452, hidden_dim=mlp_hidden_dim)

    def forward(self, tokens, graph_data):
        token_embeddings = self.fm_model(tokens,repr_layers=[12])['representations'][12]
        token_embeddings = torch.max(token_embeddings, dim=1).values 
        graph_embeddings = self.gnn(graph_data)    
        image_embeddings = torch.zeros(token_embeddings.size(0), 512, device=token_embeddings.device)
        combined = torch.cat((token_embeddings, graph_embeddings,image_embeddings), dim=1)
        return self.mlp(combined)
class RNAFM_CNN(nn.Module):
    def __init__(self, gnn_config,mlp_hidden_dim):
        super(RNAFM_CNN, self).__init__()
        data_dir = '../input/rnafm-tutorial/'
        temp_model, alphabet = fm.pretrained.rna_fm_t12(Path(data_dir, 'RNA-FM_pretrained.pth'))
    
        self.fm_model=fm.BioBertModel(temp_model.args,alphabet)
        self.cnn=ImageMol("ResNet18")

        
        self.mlp = MLP(input_dim=1452, hidden_dim=mlp_hidden_dim)

    def forward(self, tokens, image_data):
        token_embeddings = self.fm_model(tokens,repr_layers=[12])['representations'][12]
        token_embeddings = torch.max(token_embeddings, dim=1).values 
        graph_embeddings = torch.zeros(token_embeddings.size(0), 300, device=token_embeddings.device)
        image_embeddings = self.cnn(image_data)
        combined = torch.cat((token_embeddings, graph_embeddings,image_embeddings), dim=1)
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
        token_embeddings = self.fm_model(tokens,repr_layers=[12])['representations'][12] #B,T,E
        token_embeddings = torch.max(token_embeddings, dim=1).values #B,E
        graph_embeddings = self.gnn(graph_data)    
        image_embeddings = self.cnn(image_data)
        combined = torch.cat((token_embeddings, graph_embeddings,image_embeddings), dim=1)
        return self.mlp(combined)
class RNAFM_Drugchat_mean(nn.Module):
    def __init__(self, gnn_config,mlp_hidden_dim):
        super(RNAFM_Drugchat_mean, self).__init__()
        data_dir = '../input/rnafm-tutorial/'
        temp_model, alphabet = fm.pretrained.rna_fm_t12(Path(data_dir, 'RNA-FM_pretrained.pth'))
    
        self.fm_model=fm.BioBertModel(temp_model.args,alphabet)
        self.gnn = GNN_graphpred(**gnn_config)

        self.cnn=ImageMol("ResNet18")
        self.mlp = MLP(input_dim=1452, hidden_dim=mlp_hidden_dim)

    def forward(self, tokens, graph_data,image_data):
        token_embeddings = self.fm_model(tokens,repr_layers=[12])['representations'][12]
        token_embeddings = torch.mean(token_embeddings, dim=1)
        graph_embeddings = self.gnn(graph_data)    
        image_embeddings = self.cnn(image_data)
        combined = torch.cat((token_embeddings, graph_embeddings,image_embeddings), dim=1)
        return self.mlp(combined)

class RNAFM_Drugchat_MultiRow(nn.Module):
    """
    处理多行RNA序列数据的模型
    
    支持两种输入模式：
    1. 训练模式: tokens shape为 (B, T) - 单行RNA序列
    2. 测试模式: tokens shape为 (B, L, T) - 多行RNA序列
    
    处理流程(多行模式):
    1. 对每行进行RNA-FM编码得到 (B, L, T, E)
    2. 先在L维度做mean pooling得到 (B, T, E)
    3. 再在T维度做max pooling得到 (B, E)
    4. 与GNN和CNN输出concat后通过MLP
    """
    def __init__(self, gnn_config, mlp_hidden_dim):
        super(RNAFM_Drugchat_MultiRow, self).__init__()
        data_dir = '../input/rnafm-tutorial/'
        temp_model, alphabet = fm.pretrained.rna_fm_t12(Path(data_dir, 'RNA-FM_pretrained.pth'))
    
        self.fm_model = fm.BioBertModel(temp_model.args, alphabet)
        self.gnn = GNN_graphpred(**gnn_config)
        self.cnn = ImageMol("ResNet18")
        self.mlp = MLP(input_dim=1452, hidden_dim=mlp_hidden_dim)

    def forward(self, tokens, graph_data, image_data):
        """
        支持两种输入模式：
        1. 训练模式: tokens shape为 (B, T) - 单行RNA序列
        2. 测试模式: tokens shape为 (B, L, T) - 多行RNA序列
        
        注意：所有输入数据应该已经在正确的device上
        """
        # 检测输入维度，支持2D和3D tokens
        if tokens.dim() == 2:
            # 单行模式 (B, T) - 用于训练
            return self._forward_single_row(tokens, graph_data, image_data)
        elif tokens.dim() == 3:
            # 多行模式 (B, L, T) - 用于测试
            return self._forward_multi_row(tokens, graph_data, image_data)
        else:
            raise ValueError(f"tokens must be 2D (B, T) or 3D (B, L, T), got shape {tokens.shape}")
    
    def _forward_single_row(self, tokens, graph_data, image_data):
        """处理单行RNA序列 (B, T)"""
        # Get RNA-FM embeddings: (B, T, E)
        token_embeddings = self.fm_model(tokens, repr_layers=[12])['representations'][12]
        
        # Max pooling over T dimension: (B, T, E) -> (B, E)
        token_embeddings = torch.max(token_embeddings, dim=1).values
        
        # 处理 GNN
        graph_embeddings = self.gnn(graph_data)
        
        # 处理 CNN
        image_embeddings = self.cnn(image_data)
        
        # 合并并通过MLP
        combined = torch.cat((token_embeddings, graph_embeddings, image_embeddings), dim=1)
        return self.mlp(combined)
    
    def _forward_multi_row(self, tokens, graph_data, image_data):
        """处理多行RNA序列 (B, L, T)"""
        # tokens shape: (B, L, T) where B=batch, L=lines/rows, T=token_length
        B, L, T = tokens.shape
        
        # Reshape
        tokens_reshaped = tokens.view(B * L, T)
        
        # Get RNA-FM embeddings for all rows: (B*L, T, E)
        token_embeddings = self.fm_model(tokens_reshaped, repr_layers=[12])['representations'][12]
        _, T_out, E = token_embeddings.shape
        
        # Reshape back to (B, L, T, E)
        token_embeddings = token_embeddings.view(B, L, T_out, E)
        
        # Mean pooling over L dimension: (B, L, T, E) -> (B, T, E)
        token_embeddings = torch.mean(token_embeddings, dim=1)
        
        # Max pooling over T dimension: (B, T, E) -> (B, E)
        token_embeddings = torch.max(token_embeddings, dim=1).values
        
        # 处理 GNN
        graph_embeddings = self.gnn(graph_data)
        
        # 处理 CNN
        image_embeddings = self.cnn(image_data)
        
        # 合并并通过MLP
        combined = torch.cat((token_embeddings, graph_embeddings, image_embeddings), dim=1)
        return self.mlp(combined)
