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
from src.models.model import RNAFM_Drugchat 

# ------------------ Argument Parser ------------------
parser = argparse.ArgumentParser()
parser.add_argument('--train_path', type=str, required=True)
parser.add_argument('--test_path', type=str, required=True)
parser.add_argument('--output_dir', type=str, default='outputs')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--patience', type=int, default=5)
parser.add_argument('--project', type=str, default='fusion-rnafm')
parser.add_argument('--run_name', type=str, default=time.strftime('%Y%m%d-%H%M%S'))
parser.add_argument('--mlp_hidden_dim', type=int, default=1024)
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)
wandb.init(project=args.project, name=args.run_name, config=vars(args))

# ------------------ Device ------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ------------------ Load Dataset ------------------
train_data = torch.load(args.train_path)
test_data = torch.load(args.test_path)

class MultiModalDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]  # (tokens, graph, image, label)

def collate_fn(batch):
    tokens, graphs, images, labels = zip(*batch)

    tokens = torch.stack(tokens)
    pad_token_id = 1
    
    lengths = (tokens != pad_token_id).sum(dim=1)
    max_len = lengths.max()

   
    tokens = tokens[:, :max_len]
    graphs = Batch.from_data_list(graphs)
    images = torch.stack(images)
    labels = torch.tensor(labels, dtype=torch.float)
    return tokens, graphs, images, labels

val_len = len(test_data) // 2
test_len = len(test_data) - val_len
val_dataset, test_dataset = random_split(test_data, [val_len, test_len], generator=torch.Generator().manual_seed(42))

train_loader = DataLoader(MultiModalDataset(train_data), batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=collate_fn)

# ------------------ Model ------------------
gnn_config = {
    "num_layer": 5,
    "emb_dim": 300,
    "num_tasks": 300,
    "JK": "last",
    "graph_pooling": "attention",
    "gnn_type": "gin"
}

model = RNAFM_Drugchat(gnn_config=gnn_config, mlp_hidden_dim=args.mlp_hidden_dim).to(device)

# ------------------ Loss, Optimizer ------------------
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

best_val_loss = float('inf')
counter = 0
best_model_state = None
train_losses, val_losses = [], []

# ------------------ Training ------------------
for epoch in range(args.epochs):
    model.train()
    total_loss = 0
    for tokens, graphs, images, labels in train_loader:
        tokens, graphs, images, labels = tokens.to(device), graphs.to(device), images.to(device), labels.to(device)

        optimizer.zero_grad()
        output = model(tokens, graphs, images)
        loss = criterion(output, labels.float())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for tokens, graphs, images, labels in val_loader:
            tokens, graphs, images, labels = tokens.to(device), graphs.to(device), images.to(device), labels.to(device)
            output = model(tokens, graphs, images)
            loss = criterion(output, labels.float())
            val_loss += loss.item()

    total_loss /= len(train_loader)
    val_loss /= len(val_loader)
    train_losses.append(total_loss)
    val_losses.append(val_loss)

    print(f"[Epoch {epoch+1}] Train Loss: {total_loss:.4f} | Val Loss: {val_loss:.4f}")
    wandb.log({"train_loss": total_loss, "val_loss": val_loss, "epoch": epoch+1})
    scheduler.step(val_loss)

    
    path = os.path.join(args.output_dir, 'latest_model.pt')
    torch.save(model.state_dict(), path)
    wandb.save(path)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_state = model.state_dict()
        counter = 0
    else:
        counter += 1
        if counter >= args.patience:
            print("Early stopping triggered.")
            break

# ------------------ Save Best Model ------------------
if best_model_state:
    model.load_state_dict(best_model_state)
    model_path = os.path.join(args.output_dir, 'best_model.pt')
    torch.save(model.state_dict(), model_path)
    wandb.save(model_path)

# ------------------ Evaluation ------------------
model.eval()
y_true, y_pred, y_score = [], [], []

with torch.no_grad():
    for tokens, graphs, images, labels in test_loader:
        tokens, graphs, images = tokens.to(device), graphs.to(device), images.to(device)
        output = model(tokens, graphs, images)
        probs = output.cpu()
        preds = (probs > 0.5).long()
        y_true.extend(labels.tolist())
        y_pred.extend(preds.tolist())
        y_score.extend(probs.tolist())

acc = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
auc = roc_auc_score(y_true, y_score)

print(f"Test Accuracy : {acc * 100:.2f}%")
print(f"F1 Score      : {f1:.4f}")
print(f"Precision     : {precision:.4f}")
print(f"Recall        : {recall:.4f}")
print(f"AUC           : {auc:.4f}")
wandb.log({
    "test_acc": acc,
    "test_f1": f1,
    "test_precision": precision,
    "test_recall": recall,
    "test_auc": auc
})

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

plt.figure()
fpr, tpr, _ = roc_curve(y_true, y_score)
plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.legend()
roc_path = os.path.join(args.output_dir, 'roc_curve.png')
plt.savefig(roc_path)
wandb.log({"roc_curve": wandb.Image(roc_path)})

plt.figure()
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
cm_path = os.path.join(args.output_dir, 'confusion_matrix.png')
plt.savefig(cm_path)
wandb.log({"confusion_matrix": wandb.Image(cm_path)})

print(f"All results saved to {args.output_dir}")
wandb.finish()