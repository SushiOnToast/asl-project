"""
This file runs a baseline test by fitting logistic regression (with StandardScaler) and a random forest 
on the train rows, both with a class_weight="balanced" so that small classes are not ignored. Evaluation is done
on validation rows only
We get to see validation accuracy and training time per model with the 5 weakest letters by per-class accuracy
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import time

# load data
d = np.load("features_clean.npz")
X, y, _ = d["X"], d["y"], d["paths"]
# load split indexes and split data accordingly
s = np.load("splits.npz")
Xtr, ytr, Xva, yva = X[s["train"]], y[s["train"]], X[s["val"]], y[s["val"]]

print(Xtr.shape, Xva.shape) 
print(f"chance: {100/len(np.unique(y)):.1f}%")

models = {
    "Logistic regression": make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")),
    "Random forest": RandomForestClassifier(
        n_estimators=200, n_jobs=-1, class_weight="balanced", random_state=0),
}

for name, model in models.items():
    t = time.time()
    model.fit(Xtr, ytr)
    pred = model.predict(Xva)
    acc = (pred == yva).mean()
    print(f"{name}: val accuracy = {100 * acc:.2f}%  ({time.time() - t:.0f}s)")

    # Per-class accuracy, worst five
    per_class = {c: (pred[yva == c] == c).mean() for c in sorted(set(yva.tolist()))}
    worst = sorted(per_class.items(), key=lambda kv: kv[1])[:5]
    print("  weakest letters:", ", ".join(f"{c}={100 * a:.0f}%" for c, a in worst), "\n")