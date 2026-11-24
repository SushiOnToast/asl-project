"""
This file exports the final MLP for the browser version of this project
"""

import json

import numpy as np
import torch

from asl import paths
from asl.inference import load_model, predict
from asl.normalisation import normalise_batch

# this is where we will store the model for web version use
OUT = paths.ROOT.parent / "web" / "public" / "model"
OUT.mkdir(parents=True, exist_ok=True)

model, classes, mean, std = load_model(paths.MODEL_COMBINED)

# export model to onnx
torch.onnx.export(
    model,
    torch.zeros(1, 63),
    str(OUT / "asl_mlp.onnx"),
    input_names=["features"],
    output_names=["logits"],
)

# writing the other data into a metadata file (classes, mean, std)
(OUT / "metadata.json").write_text(
    json.dumps({"classes": list(classes), "mean": mean.tolist(), "std": std.tolist()})
)

# parity fixtures (tests to see if exported onnx produces identical results, will be used by vitest in web folder)
raw = np.load(paths.RAW_LANDMARKS)["X"]
rng = np.random.default_rng(0)  # random generator
cases = []

for n, i in enumerate(
    rng.choice(len(raw), 8, replace=False)
):  # pick 8 features [indexes]
    hand = raw[i].astype(np.float32).copy()
    w, h = (
        (200, 200) if n % 2 == 0 else (640, 480)
    )  # testing different aspect rations (both camera and image)
    if n == 7:
        hand = (hand[0] + (hand - hand[0]) * 0.2).astype(
            np.float32
        )  # make a tiny hand with expected result of being rejected
    probs, ok = predict(model, mean, std, hand, frame_h=h, frame_w=w)

    mat = hand.copy()
    mat[:, 1] *= h / w
    feats, size = normalise_batch(mat[None])
    cases.append(
        {
            "landmarks": hand.tolist(),
            "width": w,
            "height": h,
            "features": feats[0].tolist(),
            "size": float(size[0]),
            "valid": bool(ok),
            "probs": probs.tolist() if ok else None,
        }
    )

(OUT / "fixtures.json").write_text(json.dumps(cases))

