"""
this file contains utility functions that normalise the images by ensuring that the model
recieves sized and positioned hands for classification, also includes the functions required
for testing invariance
"""

import numpy as np

EPS = 1e-6

def normalise_hand(hand):
    """
    Given a hand matrix in the shape (21, 3), returns a normalised, flattened
    hand where points on the hand are relative to the wrist and normalised
    by hand size
    """
    wrist_relative_hand = hand - hand[0]
    hand_size = max(np.linalg.norm(wrist_relative_hand[9, :2]), EPS)
    normalised_hand = wrist_relative_hand / hand_size

    return normalised_hand.flatten()

def check_invariance(hand):
    """
    returns true if normalising hand function returns similar values
    when the hand is shifted and scaled, indicating that the normalisation
    is working
    """
    hand_copy = hand * 0.5
    hand_copy[:, :2] += 0.2

    hand_result = normalise_hand(hand)
    hand_copy_result = normalise_hand(hand_copy)

    return np.allclose(hand_result, hand_copy_result)

def check_batch_invariance(X, sizes, n=1000, min_size=0.1):
    """Shrink + shift real hands; features should not change (for non-tiny hands)."""
    moved = X * 0.5
    moved[:, :, :2] += 0.2
    f1, _ = normalise_batch(X)
    f2, _ = normalise_batch(moved)
    diff = np.abs(f1 - f2).max(axis=1)
    print("max diff, all samples:     ", diff.max())
    print("max diff, size >= MIN_SIZE:", diff[sizes >= min_size].max())

def normalise_batch(hands):
    """
    normalises a batch of hands by subtracting the wrist landmark and
    normalising for hand size
    """
    wrist_relative_batch = hands - hands[:, 0:1]

    hand_size_coords = wrist_relative_batch[:, 9, :2]

    hand_sizes = (np.linalg.norm(hand_size_coords, axis=1))
    safe = np.maximum(hand_sizes, EPS)[:, None, None]
    normalised_hands = wrist_relative_batch / safe

    return normalised_hands.reshape(len(hands), -1), hand_sizes

def mirror(hands):
    """
    Takes in a batch of normalised hands in shape (N, 63), returns mirrored hands
    in the exact same shape (it negates the x value)
    """
    h = hands.copy()
    h = h.reshape(len(hands), 21, 3)
    h[:, :, 0] *= -1

    return h.reshape(len(hands), -1)

# testing
if __name__ == "__main__":
    hands = np.random.rand(5, 21, 3).astype(np.float32)

    normalised, _ = normalise_batch(hands)

    print(f"Original and mirror(mirror(F)) similarity: {np.allclose(normalised, mirror(mirror(normalised)))}")

    