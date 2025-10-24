"""
训练脚本：RNAFM_Drugchat_MultiRow模型

数据处理策略：
- 训练阶段：将每个样本的RNA token按行展开，每一行与同一个graph、image组合作为独立样本
  输入: (B, L, T) -> 展开为 (B*L, T)，每行独立训练
  使用梯度累积：每8个展开后的样本做一次参数更新（实际batch size = 8）
  
- 验证/测试阶段：使用完整的多行RNA token输入
  输入: (B, L, T)，所有行一起进行预测

这种策略使得训练时能够充分利用每行RNA序列的信息，而测试时能够综合所有行做出预测。
"""

import os
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, roc_curve, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
from torch.utils.data import DataLoader, random_split
from torch_geometric.data import Batch
from src.models.model import RNAFM_Drugchat_MultiRow

# ------------------ Metric Function ------------------
def get_metrics(y_true, y_pred_prob, threshold=0.5):
    y_pred = (y_pred_prob > threshold).long()
    y_true_np = y_true.tolist()
    y_pred_np = y_pred.tolist()
    y_prob_np = y_pred_prob.tolist()
    return {
        "acc": accuracy_score(y_true_np, y_pred_np),
        "precision": precision_score(y_true_np, y_pred_np),
        "recall": recall_score(y_true_np, y_pred_np),
        "f1": f1_score(y_true_np, y_pred_np),
        "auc": roc_auc_score(y_true_np, y_prob_np)
    }

# ------------------ Argument Parser ------------------
parser = argparse.ArgumentParser()
parser.add_argument('--train_path', type=str, default='datasets/trainset_large.pt')
parser.add_argument('--val_path', type=str, default='datasets/valset_large.pt')
parser.add_argument('--test_path', type=str, default='datasets/testset_large.pt')
parser.add_argument('--output_dir', type=str, default='outputs')
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--patience', type=int, default=5)
parser.add_argument('--project', type=str, default='fusion-rnafm')
parser.add_argument('--run_name', type=str, default=time.strftime('%Y%m%d-%H%M%S'))
parser.add_argument('--mlp_hidden_dim', type=int, default=1024)

args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)


wandb_id_path = os.path.join(args.output_dir, 'wandb_id.txt')

wandb_id = wandb.util.generate_id()
with open(wandb_id_path, 'w') as f:
    f.write(wandb_id)

wandb.init(
    project=args.project,
    name=args.run_name,
    config=vars(args),
    id=wandb_id,

)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load datasets
print(f"Loading train data from: {args.train_path}")
train_data = torch.load(args.train_path)
print(f"Loading validation data from: {args.val_path}")
val_data = torch.load(args.val_path)
print(f"Loading test data from: {args.test_path}")
test_data = torch.load(args.test_path)

print(f"Dataset sizes - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

class MultiModalDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn(batch):
    tokens, graphs, images, labels = zip(*batch)
    tokens = torch.stack(tokens)
    graphs = Batch.from_data_list(graphs)
    images = torch.stack(images)
    labels = torch.tensor(labels, dtype=torch.float)
    return tokens, graphs, images, labels

def expand_batch_for_training(tokens, graphs, images, labels):
    """
    将batch展开用于训练：每个样本的每一行RNA token与同一个graph、image组合
    
    输入:
        tokens: (B, L, T) - B个样本，每个L行，每行T个token
        graphs: Batch对象，包含B个graph
        images: (B, C, H, W) - B个图像
        labels: (B,) - B个标签
    
    输出:
        tokens_expanded: (B*L, T) - B*L个样本，每个是单行token
        graphs_expanded: Batch对象，包含B*L个graph（每个graph重复L次）
        images_expanded: (B*L, C, H, W) - B*L个图像（每个image重复L次）
        labels_expanded: (B*L,) - B*L个标签（每个label重复L次）
    """
    B, L, T = tokens.shape
    
    # 展开tokens: (B, L, T) -> (B*L, T)
    tokens_expanded = tokens.view(B * L, T)
    
    # 复制labels: (B,) -> (B*L,)
    labels_expanded = labels.repeat_interleave(L)
    
    # 复制images: (B, C, H, W) -> (B*L, C, H, W)
    images_expanded = images.repeat_interleave(L, dim=0)
    
    # 复制graphs: 需要将每个graph重复L次
    from torch_geometric.data import Data
    graph_list = graphs.to_data_list()  # 转换为list
    expanded_graph_list = []
    for graph in graph_list:
        for _ in range(L):
            # 复制graph（深拷贝避免共享引用）
            expanded_graph_list.append(graph.clone())
    graphs_expanded = Batch.from_data_list(expanded_graph_list)
    
    return tokens_expanded, graphs_expanded, images_expanded, labels_expanded

# Create data loaders
train_loader = DataLoader(MultiModalDataset(train_data), batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(MultiModalDataset(val_data), batch_size=args.batch_size, collate_fn=collate_fn)
test_loader = DataLoader(MultiModalDataset(test_data), batch_size=args.batch_size, collate_fn=collate_fn)

# ------------------ Model ------------------
gnn_config = {
    "num_layer": 5,
    "emb_dim": 300,
    "num_tasks": 300,
    "JK": "last",
    "graph_pooling": "attention",
    "gnn_type": "gin"
}
model = RNAFM_Drugchat_MultiRow(
    gnn_config=gnn_config, 
    mlp_hidden_dim=args.mlp_hidden_dim
).to(device)

print(f"Model loaded on {device}")
print(f"Training configuration:")
print(f"  - Batch size (before expansion): {args.batch_size}")
print(f"  - Gradient accumulation: Every 8 samples (after expansion)")
print(f"  - Effective batch size: 8 samples")

# ------------------ Loss, Optimizer ------------------
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

# ------------------ Resume Checkpoint ------------------
start_epoch = 0
best_val_loss = float('inf')
counter = 0
train_losses, val_losses = [], []

for epoch in range(start_epoch, args.epochs):
    model.train()
    total_loss = 0
    optimizer.zero_grad()  # 在epoch开始时清零梯度
    
    # 梯度累积配置：每8个样本做一次参数更新
    accumulated_samples = 0  # 累积的样本数（展开后的样本数）
    target_samples = 8  # 实际batch size = 8
    
    for batch_idx, (tokens, graphs, images, labels) in enumerate(train_loader):
        # 训练阶段：展开数据，每行RNA token与graph、image独立组合
        # 例如：如果batch_size=2，每个样本有L行，展开后变成 2*L 个样本
        tokens, graphs, images, labels = expand_batch_for_training(tokens, graphs, images, labels)
        
        # 当前batch的样本数（展开后）
        current_batch_size = tokens.size(0)
        
        # 将graphs转换为list以便切片
        graph_list = graphs.to_data_list()
        
        # 严格按照8个样本为单位进行梯度累积
        # 如果当前batch展开后的样本数超过8，会分成多个sub-batch处理
        # 例如：current_batch_size=10, accumulated_samples=0, target_samples=8
        #   第一轮：处理前8个样本 (start=0, end=8)，累积到8个 → 更新参数
        #   第二轮：处理后2个样本 (start=8, end=10)，累积到2个 → 继续累积
        # 例如：current_batch_size=5, accumulated_samples=6, target_samples=8
        #   当前轮：处理前2个样本 (start=0, end=2)，累积到8个 → 更新参数
        #   下一轮：处理后3个样本 (start=2, end=5)，累积到3个 → 继续累积
        start_idx = 0
        while start_idx < current_batch_size:
            # 计算当前sub-batch可以处理多少样本
            remaining_to_target = target_samples - accumulated_samples
            end_idx = min(start_idx + remaining_to_target, current_batch_size)
            
            # 提取sub-batch并转移到device
            sub_tokens = tokens[start_idx:end_idx].to(device)
            sub_graph_list = graph_list[start_idx:end_idx]
            sub_graphs = Batch.from_data_list(sub_graph_list).to(device)
            sub_images = images[start_idx:end_idx].to(device)
            sub_labels = labels[start_idx:end_idx].to(device)
            
            # 前向传播（所有数据已在device上）
            output = model(sub_tokens, sub_graphs, sub_images)  # 输入2D tokens
            loss = criterion(output, sub_labels.float())
            
            # 反向传播（梯度累积）
            loss.backward()
            
            # 更新累积计数
            sub_batch_size = end_idx - start_idx
            accumulated_samples += sub_batch_size
            total_loss += loss.item()
            
            # 如果累积到8个样本，或者是最后一个batch的最后一个sub-batch，更新参数
            is_last_batch = (batch_idx + 1) == len(train_loader)
            is_last_sub_batch = end_idx >= current_batch_size
            
            if accumulated_samples >= target_samples or (is_last_batch and is_last_sub_batch):
                optimizer.step()  # 使用累积的梯度更新参数
                optimizer.zero_grad()  # 清零梯度，准备下一次累积
                accumulated_samples = 0  # 重置计数器
            
            start_idx = end_idx
    
    total_loss /= len(train_loader)

    model.eval()
    y_tr_true, y_tr_prob = [], []
    # with torch.no_grad():
    #     for tokens, graphs, images, labels in train_loader:
    #         # 评估训练集：使用多行模式 (B, L, T)，不展开数据
    #         # 转移数据到device
    #         tokens = tokens.to(device)
    #         graphs = graphs.to(device)
    #         images = images.to(device)
            
    #         logits = model(tokens, graphs, images)  # 输入3D tokens (B, L, T)
    #         probs = torch.sigmoid(logits).cpu()
    #         y_tr_true.extend(labels)
    #         y_tr_prob.extend(probs)
    train_metrics = get_metrics(torch.tensor(y_tr_true), torch.stack(y_tr_prob))

    val_loss = 0
    y_val_true, y_val_prob = [], []
    with torch.no_grad():
        for tokens, graphs, images, labels in val_loader:
            # 验证阶段：使用多行模式 (B, L, T)，不展开数据
            # 转移数据到device
            tokens = tokens.to(device)
            graphs = graphs.to(device)
            images = images.to(device)
            labels = labels.to(device)
            
            logits = model(tokens, graphs, images)  # 输入3D tokens (B, L, T)
            loss = criterion(logits, labels.float())
            probs = torch.sigmoid(logits)
            val_loss += loss.item()
            y_val_true.extend(labels.cpu())
            y_val_prob.extend(probs.cpu())
    val_loss /= len(val_loader)
    val_metrics = get_metrics(torch.tensor(y_val_true), torch.stack(y_val_prob))

    train_losses.append(total_loss)
    val_losses.append(val_loss)

    # 打印epoch结果
    print(f"[Epoch {epoch+1}] Train Loss: {total_loss:.4f} | Val Loss: {val_loss:.4f}")
    
    # 记录到wandb
    wandb.log({
        "epoch": epoch + 1,
        "train/loss": total_loss,
        **{f"train/{k}": v for k, v in train_metrics.items()},
        "val/loss": val_loss,
        **{f"val/{k}": v for k, v in val_metrics.items()}
    }, step=epoch + 1)

    scheduler.step(val_loss)



    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_state = model.state_dict()
        counter = 0
    else:
        counter += 1
        if counter >= args.patience:
            print("Early stopping triggered.")
            break

# Save best model
if best_model_state:
    model.load_state_dict(best_model_state)
    best_model_path = os.path.join(args.output_dir, 'best_model.pt')
    torch.save({'model_state_dict': best_model_state}, best_model_path)
    wandb.save(best_model_path)

# ------------------ Evaluation ------------------
model.eval()
y_true, y_prob = [], []
with torch.no_grad():
    for tokens, graphs, images, labels in test_loader:
        # 测试阶段：使用多行模式 (B, L, T)，不展开数据
        # 转移数据到device
        tokens = tokens.to(device)
        graphs = graphs.to(device)
        images = images.to(device)
        
        logits = model(tokens, graphs, images)  # 输入3D tokens (B, L, T)
        probs = torch.sigmoid(logits).cpu()
        y_true.extend(labels)
        y_prob.extend(probs)

test_metrics = get_metrics(torch.tensor(y_true), torch.stack(y_prob))
for k, v in test_metrics.items():
    print(f"{k:10}: {v:.4f}")
wandb.log({f"test/{k}": v for k, v in test_metrics.items()})

# ------------------ Plots ------------------
plt.figure()
plt.plot(train_losses, label='Train')
plt.plot(val_losses, label='Val')
plt.title("Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
loss_path = os.path.join(args.output_dir, 'loss_curve.png')
plt.savefig(loss_path)
wandb.log({"loss_curve": wandb.Image(loss_path)})

import csv

plt.figure()
scores = [float(x) for x in y_prob]
fpr, tpr, thresholds = roc_curve(y_true, scores)
plt.plot(fpr, tpr, label=f"AUC = {test_metrics['auc']:.4f}")
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.legend()

os.makedirs(args.output_dir, exist_ok=True)
roc_path = os.path.join(args.output_dir, 'roc_curve.png')
plt.savefig(roc_path)
plt.close()

roc_csv_path = os.path.join(args.output_dir, 'roc_curve_data.csv')
with open(roc_csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['fpr', 'tpr', 'threshold'])
    writer.writerows(zip(fpr, tpr, thresholds))

roc_table = wandb.Table(columns=['fpr', 'tpr', 'threshold'])
for row in zip(fpr, tpr, thresholds):
    roc_table.add_data(*row)

wandb.log({
    "roc_curve": wandb.Image(roc_path),
    "roc_curve_data": roc_table
})
plt.figure()
cm = confusion_matrix(y_true, [(x > 0.5).long() for x in y_prob])
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
cm_path = os.path.join(args.output_dir, 'confusion_matrix.png')
plt.savefig(cm_path)
wandb.log({"confusion_matrix": wandb.Image(cm_path)})

print(f"All results saved to {args.output_dir}")
wandb.finish()