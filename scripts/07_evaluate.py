from asl.inference import load_model
from asl import paths
import numpy as np
import csv
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import classification_report, confusion_matrix
from asl.normalisation import mirror

model, classes, mean, std = load_model(paths.MODEL_MIRROR)

# load data
d = np.load(paths.CLEAN_FEATURES)
X, y, _ = d["X"], d["y"], d["paths"]

cls_to_idx = {c: i for i, c in enumerate(classes)}
y_idx = np.array([cls_to_idx[c] for c in y])

# load split indexes and split data accordingly
s = np.load(paths.SPLITS)
Xva, yva = X[s["val"]], y_idx[s["val"]]

# standardise data using mean and std
Xva_s = (Xva - mean) / std

def predict_all(model, X_std):
    """(N, 63) standardised features -> (N,) predicted class numbers."""
    with torch.no_grad():
        logits = model(torch.from_numpy(X_std).float())
    return logits.argmax(dim=1).numpy()


def top_mistakes(cm, classes, k=12):
    off = cm.copy()
    np.fill_diagonal(off, 0)                       # ignore correct predictions
    print(f"Top {k} mistakes (true -> predicted : count):")
    for flat in np.argsort(off.ravel())[::-1][:k]:
        i, j = divmod(flat, len(classes))          # flat index -> (row, column)
        if off[i, j] > 0:
            print(f"  {classes[i]} -> {classes[j]} : {off[i, j]}")


def evaluate_set(name, X_std, y_true):
    print(f"\n{'=' * 60}\n{name}\n{'=' * 60}")
    pred = predict_all(model, X_std)
    print(f"accuracy: {100 * (pred == y_true).mean():.2f}%\n")
    print(classification_report(y_true, pred, labels=range(len(classes)),
                                target_names=classes, digits=3))
    cm = confusion_matrix(y_true, pred, labels=range(len(classes)))
    top_mistakes(cm, classes)
    return pred, cm


# normal and mirrored validation
pred, cm = evaluate_set("Normal validation", Xva_s, yva)
Xva_m_s = (mirror(Xva) - mean) / std
evaluate_set("Mirrored validation", Xva_m_s, yva)

cm_norm = cm / cm.sum(axis=1, keepdims=True)
fig, ax = plt.subplots(figsize=(10, 9))
im = ax.imshow(cm_norm, cmap="Blues")
ax.set_xticks(range(len(classes)))
ax.set_xticklabels(classes)
ax.set_yticks(range(len(classes)))
ax.set_yticklabels(classes)
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
ax.set_title("Confusion matrix (mirror model, validation)")
fig.colorbar(im, label="fraction of true letter")
plt.tight_layout()
plt.savefig(paths.ARTIFACTS / "confusion_mirror.png", dpi=150)

# save per letter metrics as baseline for later comparison
report = classification_report(yva, pred, labels=range(len(classes)),
                               target_names=classes, output_dict=True)
with open(paths.ARTIFACTS / "metrics_mirror.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["letter", "precision", "recall", "f1", "support"])
    for c in classes:
        r = report[c]
        writer.writerow([c, f"{r['precision']:.4f}", f"{r['recall']:.4f}",
                         f"{r['f1-score']:.4f}", int(r["support"])])
print("\nSaved confusion_mirror.png and metrics_mirror.csv to", paths.ARTIFACTS)

plt.show()