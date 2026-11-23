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
d = np.load(paths.CLEAN_FEATURES_ALL)
X, y, source = d["X"], d["y"], d["source"]

# we need to turn the labels into integers for pytorch
classes = sorted(set(y.tolist()))
cls_to_idx = {c: i for i, c in enumerate(classes)}
y_idx = np.array([cls_to_idx[c] for c in y])

# load split indexes and split data accordingly
s = np.load(paths.SPLITS_ALL)
tr, va = s["train"], s["val"]
Xtr, ytr = X[tr], y_idx[tr]

# split validation into sources to score separately
va_k = va[source[va] == "kaggle"]
va_h = va[source[va] == "asl_hg"]
Xva_k, yva_k = X[va_k], y_idx[va_k]
Xva_h, yva_h = X[va_h], y_idx[va_h]

# add on the mirrored hands
Xtr_aug, ytr_aug = np.concatenate([Xtr, mirror(Xtr)]), np.concatenate([ytr, ytr])
print(Xtr_aug.shape)

# standardise data using mean and std
mean = Xtr_aug.mean(axis=0) # one mean per feature (column)
std = Xtr_aug.std(axis=0) + 1e-6 # preventing 0 division error
Xtr_s = (Xtr_aug - mean) / std
Xva_sk = (Xva_k - mean) / std
Xva_sh = (Xva_h - mean) / std

# initialising dataloader
train_loader = DataLoader(LandmarkDataset(Xtr_s, ytr_aug), batch_size=256, shuffle=True)
val_k_loader = DataLoader(LandmarkDataset(Xva_sk, yva_k), batch_size=1024, shuffle=False)
val_h_loader = DataLoader(LandmarkDataset(Xva_sh, yva_h), batch_size=1024, shuffle=False)

# initialising model, weights and optimiser
model = NeuralNetwork(len(classes)).to(device)
counts = np.bincount(ytr)
weights = len(ytr_aug) / (len(classes) * counts) # rarer classes get bigger weights
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

    _, acc_k = evaluate(val_k_loader, model, device, criterion)
    loss_h, acc_h = evaluate(val_h_loader, model, device, criterion)
    print(f"epoch {epoch:>2}  train {100 * run_correct / run_n:5.2f}%  |  "
          f"val kaggle {100 * acc_k:5.2f}%  val hg {100 * acc_h:5.2f}%")

    # if this epoch accuracy increases (model better)
    if acc_h > best_acc:
        best_acc, best_epoch = acc_h, epoch
        # save model as checkpoint
        torch.save({"model": model.state_dict(), "classes": classes,
                    "mean": mean, "std": std}, paths.MODEL_COMBINED)
    # stop early if it doesn't improve after a certain amount of epochs
    elif epoch - best_epoch >= PATIENCE:
        print(f"No improvement for {PATIENCE} epochs, stopping early.")
        break

print(f"\nBest HG val accuracy: {100 * best_acc:.2f}% at epoch {best_epoch}")

# comparing old and new model on normal and mirrored validation
def score(path, X, y):
    model, classes, mean, std = load_model(path)
    assert list(classes) == classes, "class order differs from this dataset"

    Xs = (X - mean) / std  # standardise the dataset based on the own model's mean and std
    loader = DataLoader(LandmarkDataset(Xs, y), batch_size=1024, shuffle=False)

    return evaluate(loader, model, device, criterion)[1] # only interested in accuracy

print(f"\n{'':<18} {'kaggle val':>11} {'hg val':>9}")
for name, p in (("kaggle only", paths.MODEL),
                ("kaggle + mirror", paths.MODEL_MIRROR),
                ("combined", paths.MODEL_COMBINED)):
    print(f"{name:<18} {100 * score(p, Xva_k, yva_k):>10.2f}% "
          f"{100 * score(p, Xva_h, yva_h):>8.2f}%")