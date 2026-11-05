"""
Splits the cleaned features into training, validation and test data sets
In the original dataset subsequent images are frames of a video, so if we shuffle
them normally, subsequent frames would end up in testing and validation sets, which is not ideal as it means the model
would be memorising. Therefore, this program ensures that the video frames are in order so that they can be shuffled IN BLOCKS later.
"""

import re
from pathlib import Path
import numpy as np
from asl.paths import *

d = np.load(CLEAN_FEATURES)
y, paths = d["y"], d["paths"]


def frame_number(p):
    """'.../A/A57.jpg' -> 57"""
    return int(re.search(r"\d+", Path(str(p)).name).group())


train, val, test = [], [], []

for letter in np.unique(y):
    # order indexes of these specific letters in the global list
    letter_idx = np.where(y == letter)[0]                     
    frames = np.array([frame_number(p) for p in paths[letter_idx]])
    ordered = letter_idx[np.argsort(frames)] # sort the global indexes 

    # splitting letter into train, val, test
    n = len(ordered)
    a, b = int(0.70 * n), int(0.85 * n)
    train.append(ordered[:a])
    val.append(ordered[a:b])
    test.append(ordered[b:])

train, val, test = (np.concatenate(s) for s in (train, val, test)) # expand all the lists adn join together

# checks
print("sizes:", len(train), len(val), len(test), "total:", len(train) + len(val) + len(test))
print("rows in dataset:", len(y))
print("no train/val overlap:", not (set(train) & set(val)))
print("no train/test overlap:", not (set(train) & set(test)))
print("no val/test overlap:", not (set(val) & set(test)))
print("every row used once:", len(set(np.concatenate([train, val, test]))) == len(y))
print("M samples in train/val/test:",
      (y[train] == "M").sum(), (y[val] == "M").sum(), (y[test] == "M").sum())

np.savez(SPLITS, train=train, val=val, test=test)