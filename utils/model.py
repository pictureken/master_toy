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
