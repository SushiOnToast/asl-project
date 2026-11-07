"""
This file trains a model on both normal hands data and mirrored hands data so that it can identify both
It then compared this files model to the previous steps model to see the difference in accuracy

The model saved from this file is saved at paths.MODEL_MIRROR
"""

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from asl.model import LandmarkDataset, NeuralNetwork, evaluate
from asl.inference import load_model
from asl import paths
from asl.normalisation import mirror

torch.manual_seed(0)
np.random.seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

# load data
d = np.load(paths.CLEAN_FEATURES)
X, y, _ = d["X"], d["y"], d["paths"]

# we need to turn the labels into integers for pytorch
classes = sorted(set(y.tolist()))
cls_to_idx = {c: i for i, c in enumerate(classes)}
y_idx = np.array([cls_to_idx[c] for c in y])

# load split indexes and split data accordingly
s = np.load(paths.SPLITS)
Xtr, ytr, Xva, yva = X[s["train"]], y_idx[s["train"]], X[s["val"]], y_idx[s["val"]]

# add on the mirrored hands
Xtr_aug, ytr_aug = np.concatenate([Xtr, mirror(Xtr)]), np.concatenate([ytr, ytr])
print(Xtr_aug.shape)

# standardise data using mean and std
mean = Xtr_aug.mean(axis=0) # one mean per feature (column)
std = Xtr_aug.std(axis=0) + 1e-6 # preventing 0 division error
Xtr_s = (Xtr_aug - mean) / std
Xva_s = (Xva - mean) / std

# initialising dataloader
train_loader = DataLoader(LandmarkDataset(Xtr_s, ytr_aug), batch_size=256, shuffle=True)
val_loader = DataLoader(LandmarkDataset(Xva_s, yva), batch_size=1024, shuffle=False)

# initialising model, weights and optimiser
model = NeuralNetwork(len(classes)).to(device)
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

    val_loss, val_acc = evaluate(val_loader, model, device, criterion)
    print(f"epoch {epoch:>2}  train loss {run_loss / run_n:.4f} acc {100 * run_correct / run_n:5.2f}%"
          f"  |  val loss {val_loss:.4f} acc {100 * val_acc:5.2f}%")

    # if this epoch accuracy increases (model better)
    if val_acc > best_acc:
        best_acc, best_epoch = val_acc, epoch
        # save model as checkpoint
        torch.save({"model": model.state_dict(), "classes": classes,
                    "mean": mean, "std": std}, paths.MODEL_MIRROR)
    # stop early if it doesn't improve after a certain amount of epochs
    elif epoch - best_epoch >= PATIENCE:
        print(f"No improvement for {PATIENCE} epochs, stopping early.")
        break

print(f"\nBest val accuracy: {100 * best_acc:.2f}% at epoch {best_epoch}")

# comparing old and new model on normal and mirrored validation
def score(model, mean, std, X, y):
    Xs = (X - mean) / std  # standardise the dataset based on the own model's mean and std
    loader = DataLoader(LandmarkDataset(Xs, y), batch_size=1024, shuffle=False)

    return evaluate(loader, model, device, criterion)[1] # only interested in accuracy

new_model, _, new_mean, new_std = load_model(paths.MODEL_MIRROR)
old_model, _, old_mean, old_std = load_model(paths.MODEL)

Xva_m = mirror(Xva)

results = {
    "old model": (
        score(old_model, old_mean, old_std, Xva, yva),
        score(old_model, old_mean, old_std, Xva_m, yva)
    ),
    "new model": (
        score(new_model, new_mean, new_std, Xva, yva),
        score(new_model, new_mean, new_std, Xva_m, yva)
    )
}

print(f"\n{'':<10} {'normal val':>11} {'mirrored val':>13}")
for name, (a, b) in results.items():
    print(f"{name:<10} {100 * a:>10.2f}% {100 * b:>12.2f}%")
