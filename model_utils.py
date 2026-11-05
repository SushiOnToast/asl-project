import torch
from torch import nn
from torch.utils.data import Dataset
from normalisation_utils import normalise_batch

# create custom Dataset
class LandmarkDataset(Dataset):
    def __init__(self, X, y):
        super().__init__()
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.X[index], self.y[index]

# creating the network
class NeuralNetwork(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(63, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, n_classes)
        )

    def forward(self, x):
        logits = self.model(x)

        return logits

def evaluate(loader, model, device, criterion):
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            total_loss += criterion(logits, yb).item() * len(yb)
            preds = logits.argmax(dim=1)
            correct += (preds == yb).sum().item()
            n += len(yb)
    return total_loss / n, correct / n

def load_model():
    saved = torch.load("mlp_best.pt", weights_only=False)
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