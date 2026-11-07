import torch
from asl.model import NeuralNetwork
from asl.normalisation import normalise_batch

def load_model(model_path):
    saved = torch.load(model_path, weights_only=False)
    model = NeuralNetwork(len(saved["classes"]))
    model.load_state_dict(saved["model"])
    model.eval()

    return model, saved["classes"], saved["mean"], saved["std"]

def predict(model, mean, std, landmark_mat, frame_h, frame_w):

    # normalise hand
    mat = landmark_mat.copy()
    mat[:, 1] *= frame_h / frame_w # need to change some values so that the hand isn't passed in "squashed"

    features, s = normalise_batch(mat[None])
    size = s[0]

    if size < 0.1 or abs(features).max() > 5:
        return (None, False)

    # standardise
    features_s = (features - mean) / std
    features_t = torch.from_numpy(features_s).float()

    with torch.no_grad():
        logits = model(features_t)
        probs = torch.softmax(logits, dim=1)

    return probs[0].numpy(), True