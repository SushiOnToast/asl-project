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


