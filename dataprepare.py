import fm
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from torch_geometric.data import Data
import sys
import os
from rdkit import Chem
from rdkit.Chem import Draw
from PIL import Image
import io
import matplotlib.pyplot as plt
from torchvision import transforms

# 添加路径以导入smiles2graph
sys.path.append('./dataset')
from smiles2graph import smiles2graph



def Smiles2Img(smis, size=224):
    mol = Chem.MolFromSmiles(smis)
    # Draw.MolsToGridImage 直接返回 PIL Image 对象
    img = Draw.MolsToGridImage([mol], molsPerRow=1, subImgSize=(size, size))
    # 确保是 RGB 模式
    if img.mode != 'RGB':
        img = img.convert('RGB')
    return img


# 定义图像转换器
transform = transforms.Compose([
    transforms.ToTensor(),  # 转换为tensor (C, H, W)
])

# 加载alphabet（不需要加载完整模型）
print("加载alphabet...")
model_path = './RNA-FM/pretrained/RNA-FM_pretrained.pth'  # 根据你的实际路径修改
fm_model, alphabet = fm.pretrained.rna_fm_t12(model_path)
batch_converter = alphabet.get_batch_converter()

# 读取CSV文件
csv_file = 'val_text.csv'
df = pd.read_csv(csv_file)

print(f"总共有 {len(df)} 条数据")
print(f"CSV列名: {df.columns.tolist()}")

# 定义RNA序列分段函数
def split_rna_sequence(rna_seq, max_length=1022):
    """
    将RNA序列按照max_length分段
    
    参数:
        rna_seq: RNA序列字符串
        max_length: 每段的最大长度，默认1022
                   注意：加上<cls>和<eos>后，实际token长度为max_length+2
    
    返回:
        segments: 分段后的序列列表
    """
    segments = []
    seq_len = len(rna_seq)
    
    if seq_len <= max_length:
        # 序列长度小于等于max_length，不需要分段
        segments.append(rna_seq)
    else:
        # 将序列分段
        for i in range(0, seq_len, max_length):
            segment = rna_seq[i:i + max_length]
            segments.append(segment)
    
    return segments

# 用于统计
success_count = 0
fail_count = 0
total_segments = 0

# 创建数据集列表
dataset = []

print(f"\n开始逐行处理RNA序列和SMILES...")

for idx, row in tqdm(df.iterrows(), total=len(df)):
    rna_id = row['RNA_ID']
    compound_id = row['Compound_ID']
    rna_seq = row['RNA_seq']
    smiles = row['SMILES']
    label = row['label']
    
    # 1. 处理RNA序列 - 分段并转换为tokens
    rna_segments = split_rna_sequence(rna_seq, max_length=1022)
    
    # 2. 处理SMILES - 转换为graph和image
    try:
        # 2.1 转换为graph
        graph_dict = smiles2graph(str(smiles))
        
        # 创建PyTorch Geometric Data对象
        graph_data = Data(
            x=torch.tensor(graph_dict['node_feat'], dtype=torch.long),
            edge_index=torch.tensor(graph_dict['edge_index'], dtype=torch.long),
            edge_attr=torch.tensor(graph_dict['edge_feat'], dtype=torch.long),
            batch=torch.zeros(graph_dict['num_nodes'], dtype=torch.long),
            num_graphs=1
        )
        
        # 2.2 转换为image
        img = Smiles2Img(str(smiles), size=224)
        # 转换为RGB tensor
        tensor_image = transform(img)  # shape: [3, 224, 224]
        
        # 3. 一次性批量处理所有RNA分段
        # 准备所有分段的数据
        data = [(f"{rna_id}_seg{i}", segment) for i, segment in enumerate(rna_segments)]
        
        # 一次性转换所有分段为tokens
        batch_labels, batch_strs, batch_tokens = batch_converter(data)
        
        # 4. 按照baseline_vanllia格式保存: (tokens_feat, graph_data, image_data, label)
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        # 添加到数据集列表中
        dataset.append((batch_tokens, graph_data, tensor_image, label_tensor))
        
        success_count += 1
        total_segments += len(rna_segments)
        
    except Exception as e:
        print(f"\n处理失败 (行{idx}, RNA_ID={rna_id}, Compound_ID={compound_id})")
        print(f"错误: {e}")
        fail_count += 1
        continue

# 保存整个数据集
print("\n保存数据集...")
torch.save(dataset, 'datasets/valset_large.pt')

# 显示处理统计
print("\n=== 处理完成！===")
print(f"原始数据: {len(df)} 行")
print(f"成功处理: {success_count} 行")
print(f"失败数量: {fail_count} 行")
print(f"总分段数: {total_segments} 个")
print(f"平均每行分段数: {total_segments / max(1, success_count):.2f}")
print(f"数据集已保存至: datasets/val_large.pt")

