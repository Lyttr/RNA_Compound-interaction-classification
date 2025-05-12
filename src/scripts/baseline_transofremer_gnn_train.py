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
from src.models.model import TransformerGNN

# ------------------ Argument Parser ------------------
parser = argparse.ArgumentParser()
parser.add_argument('--train_path', type=str, required=True)
parser.add_argument('--test_path', type=str, required=True)
parser.add_argument('--output_dir', type=str, default='outputs')
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--epochs', type=int, default=80)
parser.add_argument('--patience', type=int, default=5)
parser.add_argument('--project', type=str, default='fusion-training')
parser.add_argument('--run_name', type=str, default=time.strftime('%Y%m%d-%H%M%S'))
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)
wandb.init(project=args.project, name=args.run_name, config=vars(args))

# ------------------ Device ------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ------------------ Load Dataset ------------------
train_data = torch.load(args.train_path)
test_data = torch.load(args.test_path)

class TensorGraphDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]  # (token_tensor, graph_data, label)

def collate_fn(batch):
    token_batch, graph_batch, label_batch = zip(*batch)
    token_batch = torch.stack(token_batch)
    graph_batch = Batch.from_data_list(graph_batch)
    label_batch = torch.stack(label_batch)
    return token_batch, graph_batch, label_batch

val_len = len(test_data) // 2
test_len = len(test_data) - val_len
val_dataset, test_dataset = random_split(test_data, [val_len, test_len], generator=torch.Generator().manual_seed(42))

train_loader = DataLoader(TensorGraphDataset(train_data), batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=collate_fn)


vocab_size = 23  
embed_size = 512   
num_heads = 8       
num_layers = 6      
gnn_config = {
    "num_layer": 5,
    "emb_dim": 300,
    "num_tasks": 300,
    "JK": "last",
    "graph_pooling": "attention",
    "gnn_type": "gin"
}
mlp_hidden_dim = 1024

model = TransformerGNN(vocab_size, embed_size, num_heads, num_layers, gnn_config, mlp_hidden_dim).to(device)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

best_val_loss = float('inf')
counter = 0
best_model_state = None
train_losses, val_losses = [], []

# ------------------ Training Loop ------------------
for epoch in range(args.epochs):
    model.train()
    total_loss = 0
    for tokens, graphs, labels in train_loader:
        tokens, labels = tokens.to(device), labels.to(device)
        graphs = graphs.to(device)

        optimizer.zero_grad()
        output = model(tokens, graphs)
        loss = criterion(output, labels.float())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for tokens, graphs, labels in val_loader:
            tokens, labels = tokens.to(device), labels.to(device)
            graphs = graphs.to(device)
            output = model(tokens, graphs)
            
            loss = criterion(output, labels.float())
            val_loss += loss.item()

    total_loss /= len(train_loader)
    val_loss /= len(val_loader)
    train_losses.append(total_loss)
    val_losses.append(val_loss)

    print(f"Epoch {epoch+1}, Train Loss: {total_loss:.4f}, Val Loss: {val_loss:.4f}")
    wandb.log({"train_loss": total_loss, "val_loss": val_loss, "epoch": epoch+1})
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

# ------------------ Save Best Model ------------------
if best_model_state:
    model.load_state_dict(best_model_state)
    model_path = os.path.join(args.output_dir, 'best_fusion_model.pt')
    torch.save(model.state_dict(), model_path)
    print(f"Best model saved to {model_path}")
    wandb.save(model_path)

# ------------------ Evaluation ------------------
model.eval()
y_true, y_pred, y_score = [], [], []
with torch.no_grad():
    for tokens, graphs, labels in test_loader:
        tokens = tokens.to(device)
        graphs = graphs.to(device)
        output = model(tokens, graphs)
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
wandb.log({"test_acc": acc, "test_f1": f1, "test_precision": precision, "test_recall": recall, "test_auc": auc})

# ------------------ Plotting ------------------
plt.figure()
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training & Validation Loss Curve')
plt.legend()
plt.grid(True)
loss_curve_path = os.path.join(args.output_dir, 'loss_curve.png')
plt.savefig(loss_curve_path)
wandb.log({"loss_curve": wandb.Image(loss_curve_path)})

plt.figure()
fpr, tpr, _ = roc_curve(y_true, y_score)
plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid(True)
roc_curve_path = os.path.join(args.output_dir, 'roc_curve.png')
plt.savefig(roc_curve_path)
wandb.log({"roc_curve": wandb.Image(roc_curve_path)})

plt.figure()
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
confusion_matrix_path = os.path.join(args.output_dir, 'confusion_matrix.png')
plt.savefig(confusion_matrix_path)
wandb.log({"confusion_matrix": wandb.Image(confusion_matrix_path)})

print(f"All results saved to {args.output_dir}")
wandb.finish()