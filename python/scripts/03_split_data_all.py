"""
Carries out data splitting functionality for train, val, and test data sets for both asl_hg and grassknoted dataset,
combines them afterwards and saves in new npz file.
"""

import re 
from pathlib import Path
import numpy as np
from asl import paths

d = np.load(paths.CLEAN_FEATURES_ALL)
y, img_paths, source, signer = d["y"], d["paths"], d["source"], d["signer"]

# we pass in volunteers as training blocks
HG_TRAIN, HG_VAL, HG_TEST = {1, 2, 3, 4, 5, 6, 7}, {8, 9}, {10}

def frame_number(p):
    return int(re.search(r"\d+", Path(str(p)).name).group())


train, val, test = [], [], []

# same splitting setup for the grassknoted dataset:
for letter in np.unique(y):
    # order indexes of these specific letters in the global list
    letter_idx = np.where((source == "kaggle") & (y == letter))[0]                     
    frames = np.array([frame_number(p) for p in img_paths[letter_idx]])
    ordered = letter_idx[np.argsort(frames)] # sort the global indexes 

    # splitting letter into train, val, test
    n = len(ordered)
    a, b = int(0.70 * n), int(0.85 * n)
    train.append(ordered[:a])
    val.append(ordered[a:b])
    test.append(ordered[b:])

hg = source == "asl_hg"
train.append(np.where(hg & np.isin(signer, list(HG_TRAIN)))[0])
val.append(np.where(hg & np.isin(signer, list(HG_VAL)))[0])
test.append(np.where(hg & np.isin(signer, list(HG_TEST)))[0])

train, val, test = (np.concatenate(s) for s in (train, val, test))

np.savez(paths.SPLITS_ALL, train=train, val=val, test=test)