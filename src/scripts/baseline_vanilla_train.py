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
parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

# -------- wandb resume --------
wandb_id_path = os.path.join(args.output_dir, 'wandb_id.txt')
if args.resume is not None and os.path.exists(wandb_id_path):
    with open(wandb_id_path, 'r') as f:
        wandb_id = f.read().strip()
    print("resume")
else:
    wandb_id = wandb.util.generate_id()
    with open(wandb_id_path, 'w') as f:
        f.write(wandb_id)

wandb.init(
    project=args.project,
    name=args.run_name,
    config=vars(args),
    id=wandb_id,
    resume='allow'
)

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
        return self.data[idx]

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

# ------------------ Resume Checkpoint ------------------
start_epoch = 0
best_val_loss = float('inf')
counter = 0
train_losses, val_losses = [], []

if args.resume is not None and os.path.exists(args.resume):
    print(f"Resuming from checkpoint: {args.resume}")
    checkpoint = torch.load(args.resume)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    best_val_loss = checkpoint.get('best_val_loss', float('inf'))
    counter = checkpoint.get('counter', 0)
    start_epoch = checkpoint.get('epoch', 0) + 1

# ------------------ Training ------------------
for epoch in range(start_epoch, args.epochs):
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
    total_loss /= len(train_loader)

    model.eval()
    y_tr_true, y_tr_prob = [], []
    with torch.no_grad():
        for tokens, graphs, images, labels in train_loader:
            tokens, graphs, images = tokens.to(device), graphs.to(device), images.to(device)
            probs = model(tokens, graphs, images).cpu()
            y_tr_true.extend(labels)
            y_tr_prob.extend(probs)
    train_metrics = get_metrics(torch.tensor(y_tr_true), torch.stack(y_tr_prob))

    val_loss = 0
    y_val_true, y_val_prob = [], []
    with torch.no_grad():
        for tokens, graphs, images, labels in val_loader:
            tokens, graphs, images, labels = tokens.to(device), graphs.to(device), images.to(device), labels.to(device)
            probs = model(tokens, graphs, images)
            loss = criterion(probs, labels.float())
            val_loss += loss.item()
            y_val_true.extend(labels.cpu())
            y_val_prob.extend(probs.cpu())
    val_loss /= len(val_loader)
    val_metrics = get_metrics(torch.tensor(y_val_true), torch.stack(y_val_prob))

    train_losses.append(total_loss)
    val_losses.append(val_loss)

    print(f"[Epoch {epoch+1}] Train Loss: {total_loss:.4f} | Val Loss: {val_loss:.4f}")
    wandb.log({
        "epoch": epoch + 1,
        "train/loss": total_loss,
        **{f"train/{k}": v for k, v in train_metrics.items()},
        "val/loss": val_loss,
        **{f"val/{k}": v for k, v in val_metrics.items()}
    }, step=epoch + 1)

    scheduler.step(val_loss)

    # Save latest checkpoint
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_val_loss': best_val_loss,
        'counter': counter
    }
    ckpt_path = os.path.join(args.output_dir, 'latest_model.pt')
    torch.save(checkpoint, ckpt_path)
    wandb.save(ckpt_path)

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
        tokens, graphs, images = tokens.to(device), graphs.to(device), images.to(device)
        probs = model(tokens, graphs, images).cpu()
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

plt.figure()
fpr, tpr, _ = roc_curve(y_true, [x.item() for x in y_prob])
plt.plot(fpr, tpr, label=f"AUC = {test_metrics['auc']:.4f}")
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.legend()
roc_path = os.path.join(args.output_dir, 'roc_curve.png')
plt.savefig(roc_path)
wandb.log({"roc_curve": wandb.Image(roc_path)})

plt.figure()
cm = confusion_matrix(y_true, [(x > 0.5).long() for x in y_prob])
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
cm_path = os.path.join(args.output_dir, 'confusion_matrix.png')
plt.savefig(cm_path)
wandb.log({"confusion_matrix": wandb.Image(cm_path)})

print(f"All results saved to {args.output_dir}")
wandb.finish()