"""
This script does a zero-shot evaluation using the most recent model up till this point (the mirro model) to see how well it does on the newly imported
asl_hg dataset.
"""

from asl.inference import load_model
from asl import paths
import numpy as np
import torch

d = np.load(paths.CLEAN_FEATURES_HG)
X, y, signer = d["X"], d["y"], d["signer"]

model, classes, mean, std = load_model(paths.MODEL_MIRROR)
y_idx = np.array([classes.index(c) for c in y])

with torch.no_grad():
    logits = model(torch.from_numpy((X - mean)/std).float())
pred = logits.argmax(dim=1).numpy()

correct = pred == y_idx

print(f"\n{'=' * 60}\nmirror model\n{'=' * 60}")
print(f"accuracy on samples:  {100 * correct.mean():.2f}%")

print("\nper letter:")
for c in classes:
    m = y == c
    print(f"  {c}: {100 * correct[m].mean():5.1f}%")