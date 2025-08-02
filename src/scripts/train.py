import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, roc_curve, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
from src.models.model import MLP
import time

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

parser = argparse.ArgumentParser()
parser.add_argument('--train_path', type=str, required=True, help='Path to training set (.pt)')
parser.add_argument('--test_path', type=str, required=True, help='Path to test set (.pt)')
parser.add_argument('--output_dir', type=str, default='outputs', help='Directory to save results')
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--epochs', type=int, default=80)
parser.add_argument('--patience', type=int, default=5)
parser.add_argument('--project', type=str, default='mlp-training', help='wandb project name')
parser.add_argument('--run_name', type=str, default=time.strftime('%Y%m%d-%H%M%S'), help='WandB run name')
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)
wandb.init(project=args.project, name=args.run_name, config=vars(args))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

train = torch.load(args.train_path)
test = torch.load(args.test_path)

X_train = torch.stack([item[0].detach() for item in train]).squeeze(1).float()
y_train = torch.stack([item[1] for item in train])
X_test = torch.stack([item[0].detach() for item in test]).squeeze(1).float()
y_test = torch.stack([item[1] for item in test])

val_len = len(X_test) // 2
test_len = len(X_test) - val_len
val_dataset, test_dataset = random_split(TensorDataset(X_test, y_test), [val_len, test_len], generator=torch.Generator().manual_seed(42))

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=4, pin_memory=True)

input_dim = X_train.shape[-1]
model = MLP(input_dim).to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

best_val_loss = float('inf')
counter = 0
best_model_state = None
train_losses, val_losses = [], []

for epoch in range(args.epochs):
    model.train()
    total_loss = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels.float())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    total_loss /= len(train_loader)

    model.eval()
    y_tr_true, y_tr_prob = [], []
    with torch.no_grad():
        for inputs, labels in train_loader:
            inputs = inputs.to(device, non_blocking=True)
            probs = model(inputs).cpu()
            y_tr_true.extend(labels)
            y_tr_prob.extend(probs)
    train_metrics = get_metrics(torch.tensor(y_tr_true), torch.stack(y_tr_prob))

    val_loss, y_val_true, y_val_prob = 0, [], []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            probs = model(inputs)
            loss = criterion(probs, labels.float())
            val_loss += loss.item()
            y_val_true.extend(labels.cpu())
            y_val_prob.extend(probs.cpu())
    val_loss /= len(val_loader)
    val_metrics = get_metrics(torch.tensor(y_val_true), torch.stack(y_val_prob))

    train_losses.append(total_loss)
    val_losses.append(val_loss)

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

if best_model_state:
    model.load_state_dict(best_model_state)
    best_model_path = os.path.join(args.output_dir, 'best_model.pt')
    torch.save(model.state_dict(), best_model_path)
    print(f"Best model saved to {best_model_path}")
    wandb.save(best_model_path)

model.eval()
y_true, y_prob = [], []
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device, non_blocking=True)
        probs = model(inputs).cpu()
        y_true.extend(labels)
        y_prob.extend(probs)

test_metrics = get_metrics(torch.tensor(y_true), torch.stack(y_prob))
for k, v in test_metrics.items():
    print(f"{k:10}: {v:.4f}")
wandb.log({f"test/{k}": v for k, v in test_metrics.items()})

plt.figure()
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training & Validation Loss Curve')
plt.legend()
plt.grid(True)
plt.tight_layout()
loss_curve_path = os.path.join(args.output_dir, 'loss_curve.png')
plt.savefig(loss_curve_path)

plt.figure()
fpr, tpr, _ = roc_curve(y_true, [x.item() for x in y_prob])
plt.plot(fpr, tpr, label=f"AUC = {test_metrics['auc']:.4f}")
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid(True)
plt.tight_layout()
roc_curve_path = os.path.join(args.output_dir, 'roc_curve.png')
plt.savefig(roc_curve_path)
wandb.log({"roc_curve": wandb.Image(roc_curve_path)})

plt.figure()
cm = confusion_matrix(y_true, [(x > 0.5).long() for x in y_prob])
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.tight_layout()
confusion_matrix_path = os.path.join(args.output_dir, 'confusion_matrix.png')
plt.savefig(confusion_matrix_path)
wandb.log({"confusion_matrix": wandb.Image(confusion_matrix_path)})

print(f"All results saved to {args.output_dir}")
wandb.finish()