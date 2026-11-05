import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

torch.manual_seed(0)
np.random.seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

# load data
d = np.load("features_clean.npz")
X, y, _ = d["X"], d["y"], d["paths"]

# we need to turn the labels into integers for pytorch
classes = sorted(set(y.tolist()))
cls_to_idx = {c: i for i, c in enumerate(classes)}
y_idx = np.array([cls_to_idx[c] for c in y])

# load split indexes and split data accordingly
s = np.load("splits.npz")
Xtr, ytr, Xva, yva = X[s["train"]], y_idx[s["train"]], X[s["val"]], y_idx[s["val"]]

# standardise data using mean and std
mean = Xtr.mean(axis=0) # one mean per feature (column)
std = Xtr.std(axis=0) + 1e-6 # preventing 0 division error
Xtr_s = (Xtr - mean) / std
Xva_s = (Xva - mean) / std

# create custom Dataset
class LandmarkDataset(Dataset):
    def __init__(self, X, y):
        super().__init__()
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.X[index], self.y[index]

# creating the network
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(63, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, len(classes))
        )

    def forward(self, x):
        logits = self.model(x)

        return logits

def evaluate(loader):
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            total_loss += criterion(logits, yb).item() * len(yb)
            preds = logits.argmax(dim=1)
            correct += (preds == yb).sum().item()
            n += len(yb)
    return total_loss / n, correct / n

# initialising dataloader
train_loader = DataLoader(LandmarkDataset(Xtr_s, ytr), batch_size=256, shuffle=True)
val_loader = DataLoader(LandmarkDataset(Xva_s, yva), batch_size=1024, shuffle=False)

# initialising model, weights and optimiser
model = NeuralNetwork().to(device)
counts = np.bincount(ytr)
weights = len(ytr) / (len(classes) * counts) # rarer classes get bigger weights
criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32).to(device))
optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)

EPOCHS = 60
PATIENCE = 10

best_acc, best_epoch = 0.0, 0
for epoch in range(1, EPOCHS + 1):
    model.train()
    run_loss, run_correct, run_n = 0.0, 0, 0

    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)

        # loop
        optimiser.zero_grad() # clear old gradients
        logits = model(xb) # forward pass
        loss = criterion(logits, yb) # cost computation
        loss.backward() # compute gradient with backprop
        optimiser.step() # update weights by taking a step in that direction

        run_loss += loss.item() * len(yb)
        run_correct += (logits.argmax(dim=1) == yb).sum().item()
        run_n += len(yb)

    val_loss, val_acc = evaluate(val_loader)
    print(f"epoch {epoch:>2}  train loss {run_loss / run_n:.4f} acc {100 * run_correct / run_n:5.2f}%"
          f"  |  val loss {val_loss:.4f} acc {100 * val_acc:5.2f}%")

    # if this epoch accuracy increases (model better)
    if val_acc > best_acc:
        best_acc, best_epoch = val_acc, epoch
        # save model as checkpoint
        torch.save({"model": model.state_dict(), "classes": classes,
                    "mean": mean, "std": std}, "mlp_best.pt")
    # stop early if it doesn't improve after a certain amount of epochs
    elif epoch - best_epoch >= PATIENCE:
        print(f"No improvement for {PATIENCE} epochs, stopping early.")
        break

print(f"\nBest val accuracy: {100 * best_acc:.2f}% at epoch {best_epoch}")


