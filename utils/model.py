import torch.nn.functional as F
from torch import nn


class NeuralNet(nn.Module):
    def __init__(self, in_features, hidden_size, out_features, batch_normalize):
        super(NeuralNet, self).__init__()
        self.bn = batch_normalize
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(in_features, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, out_features)

    def forward(self, x):
        if self.bn:
            out = self.bn1(self.fc1(x))
            out = self.relu(out)
            out = self.fc2(out)
        else:
            out = self.relu(self.fc1(x))
            out = self.fc2(out)
        return out


class FCNet(nn.Module):
    # default hidden_size=64 reference https://github.com/somepago/dbViz/blob/main/models/fcnet.py
    def __init__(self, in_features, hidden_size, out_features):
        super(FCNet, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_size * 16)
        self.fc2 = nn.Linear(hidden_size * 16, hidden_size * 8)
        self.fc3 = nn.Linear(hidden_size * 8, hidden_size * 2)
        self.fc4 = nn.Linear(hidden_size * 2, hidden_size)
        self.fc5 = nn.Linear(hidden_size, out_features)

    def forward(self, x):
        out = x.view(x.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))
        out = F.relu(self.fc4(out))
        out = self.fc5(out)
        return out
