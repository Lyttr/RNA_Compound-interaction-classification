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
    输入: tokens shape为 (B, L, T), 其中B=batch size, L=行数, T=token length
    处理流程:
    1. 对每行进行RNA-FM编码得到 (B, L, T, E)
    2. 先在L维度做mean pooling得到 (B, T, E)
    3. 再在T维度做max pooling得到 (B, E)
    4. 与GNN和CNN输出concat后通过MLP
    
    显存优化:
    - use_gradient_checkpointing: 启用梯度检查点，降低50-70%显存
    - chunk_size: 分块处理多行数据，避免一次性处理B*L
    """
    def __init__(self, gnn_config, mlp_hidden_dim, use_gradient_checkpointing=False, chunk_size=None):
        super(RNAFM_Drugchat_MultiRow, self).__init__()
        data_dir = '../input/rnafm-tutorial/'
        temp_model, alphabet = fm.pretrained.rna_fm_t12(Path(data_dir, 'RNA-FM_pretrained.pth'))
    
        self.fm_model = fm.BioBertModel(temp_model.args, alphabet)
        self.gnn = GNN_graphpred(**gnn_config)
        self.cnn = ImageMol("ResNet18")
        self.mlp = MLP(input_dim=1452, hidden_dim=mlp_hidden_dim)
        
        # 梯度检查点配置
        self.use_gradient_checkpointing = use_gradient_checkpointing
        if use_gradient_checkpointing and hasattr(self.fm_model, 'gradient_checkpointing_enable'):
            self.fm_model.gradient_checkpointing_enable()
        
        # 分块处理配置
        self.chunk_size = chunk_size

    def forward(self, tokens, graph_data, image_data):
        """
        在模型内部管理数据转移到GPU，实现更精细的显存控制
        输入可以是CPU或GPU上的数据
        """
        # 获取模型所在的device
        device = next(self.parameters()).device
        
        # tokens shape: (B, L, T) where B=batch, L=lines/rows, T=token_length
        B, L, T = tokens.shape
        
        # Step 1: 处理 RNA tokens (按需转移到GPU)
        # 如果启用分块处理，避免一次性处理 B*L 的大batch
        if self.chunk_size is not None and L > self.chunk_size:
            # 分块处理每个样本的多行数据
            all_embeddings = []
            for b in range(B):
                row_embeddings = []
                for chunk_start in range(0, L, self.chunk_size):
                    chunk_end = min(chunk_start + self.chunk_size, L)
                    # 只转移当前chunk到GPU
                    chunk_tokens = tokens[b, chunk_start:chunk_end, :].to(device)
                    
                    # 处理当前chunk: (chunk_size, T) -> (chunk_size, T, E)
                    print(chunk_tokens.shape)
                    chunk_emb = self.fm_model(chunk_tokens, repr_layers=[12])['representations'][12]
                    row_embeddings.append(chunk_emb.cpu())  # 立即移回CPU节省显存
                    
                    # 清理GPU显存
                    del chunk_tokens
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
                
                # 合并所有chunks并转移到GPU: (L, T, E)
                sample_embeddings = torch.cat(row_embeddings, dim=0).to(device)
                # Mean pooling over L: (L, T, E) -> (T, E)
                sample_embeddings = torch.mean(sample_embeddings, dim=0)
                # Max pooling over T: (T, E) -> (E,)
                sample_embeddings = torch.max(sample_embeddings, dim=0).values
                all_embeddings.append(sample_embeddings)
            
            token_embeddings = torch.stack(all_embeddings)  # (B, E)
        else:
            # 原始处理方式：一次性处理所有行
            tokens = tokens.to(device)  # 转移到GPU
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
        
        # Step 2: 处理 GNN (转移graph数据到GPU)
        graph_data = graph_data.to(device)
        graph_embeddings = self.gnn(graph_data)
        
        # Step 3: 处理 CNN (转移image数据到GPU)
        image_data = image_data.to(device)
        image_embeddings = self.cnn(image_data)
        
        # Step 4: 合并并通过MLP
        combined = torch.cat((token_embeddings, graph_embeddings, image_embeddings), dim=1)
        return self.mlp(combined)
