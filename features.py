# this file normalises the images by ensuring that the model
# recieves uniformly sized and positioned hands for classification
# this is the processing that goes on to generate these features.
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

def test_invariance(hand):
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

if __name__ == "__main__":
    hands = np.random.rand(5, 21, 3).astype(np.float32)

    # invariance: shrink + shift must not change the features
    moved = hands * 0.5
    moved[:, :, :2] += 0.2
    f1, _ = normalise_batch(hands)
    f2, _ = normalise_batch(moved)
    print("invariant:", np.allclose(f1, f2, atol=1e-5))

    # batch matches single-hand version
    print("batch == single:", np.allclose(f1[2], normalise_hand(hands[2]), atol=1e-6))
    print("shape:", f1.shape)

    