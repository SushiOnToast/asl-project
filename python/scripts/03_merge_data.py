"""
Merging data from multiple datasets for convenience so that they can be split easily and stay consistent for training.
"""

import numpy as np
from asl import paths

# first we have to merge all the data together to make it easier to split
k = np.load(paths.CLEAN_FEATURES)
h = np.load(paths.CLEAN_FEATURES_HG)

X = np.concatenate([k["X"], h["X"]])
y = np.concatenate([k["y"], h["y"]])
img_paths = np.concatenate([k["paths"], h["paths"]])
sizes = np.concatenate([k["sizes"], h["sizes"]])
source = np.concatenate([np.full(len(k["X"]), "kaggle"), h["source"]])
signer = np.concatenate([np.zeros(len(k["X"]), dtype=int), h["signer"]])

# saving here
np.savez_compressed(paths.CLEAN_FEATURES_ALL, X=X, y=y, paths=img_paths, sizes=sizes, source=source, signer=signer)
print(X.shape, np.unique(source, return_counts=True))