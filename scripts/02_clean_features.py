"""
This normalises the raw landmarks (raw_landmarks.npz) extracted from extract_landmarks.py and drops unreliable samples
It also prints diagnostics like percentiles and extreme feature counts, and their corresponsing letters, additionally performing an
invariance test.

It drops the following types of samples:
- If the hand size is below the MIN_SIZE, this is where the wrist and knuckle nearly overlap and so
dividing by the size inflates all the other coordinates and noise from rounding by a lot
- If any feature is larger than the MAX_FEATURE, because this is physically implausible, fingertips should generally
be within 2-3 hand-sizes of the wrist

After this, the filtered features are saved into features_clean.npz
"""

import numpy as np
from asl.normalisation import normalise_batch, check_batch_invariance
from asl.paths import *

MIN_SIZE = 0.1 # we want to drop hands with a size smaller than this otherwise itll blow up the differences
MAX_FEATURE = 5.0 # drop hands where any features are larger than this


def main():
    # load and separate data
    d = np.load(RAW_LANDMARKS)
    X, y, paths = d["X"], d["y"], d["paths"]
    print("raw:", X.shape)

    # batch normalise thehands
    F, sizes = normalise_batch(X)       
    print("features:", F.shape)
    print("wrist columns all zero:", np.all(F[:, :3] == 0)) # checking if its been normalised properly

    print("\nhand size percentiles:")
    for p in (0, 1, 5, 50, 95, 100):
        print(f"  {p:>3}th: {np.percentile(sizes, p):.4f}")

    worst = np.abs(F).max(axis=1)  # biggest feature in each sample
    print("\nmax feature overall:", worst.max())
    print("samples with a feature > 5:", (worst > 5).sum())

    bad = worst > MAX_FEATURE
    # we want to see if there is a correlation between bad samples and the hand size
    print("median size, bad samples:", np.median(sizes[bad]).round(3),
          "| overall:", np.median(sizes).round(3))
    letters, counts = np.unique(y[bad], return_counts=True)
    print("bad samples by letter:", dict(zip(letters.tolist(), counts.tolist())))

    check_batch_invariance(X, sizes, min_size=MIN_SIZE)

    # filter out all the bad samples
    keep = (sizes >= MIN_SIZE) & (worst <= MAX_FEATURE)
    print(f"\ndropped {(~keep).sum()} of {len(keep)}")
    for c in np.unique(y):
        in_class = (y == c)
        dropped = (in_class & ~keep).sum()
        print(f"  {c}: {dropped:>4} of {in_class.sum()}")

    print("max feature after filtering:", np.abs(F[keep]).max())

    # save to new features array
    np.savez_compressed(CLEAN_FEATURES,
                        X=F[keep], y=y[keep], paths=paths[keep], sizes=sizes[keep])
    print("saved features_clean.npz:", F[keep].shape)


if __name__ == "__main__":
    main()