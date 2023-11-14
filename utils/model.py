from torch import nn


class NeuralNet(nn.Module):
    def __init__(self, in_features, hidden_size, out_features):
        super(NeuralNet, self).__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(in_features, hidden_size)
        self.fc2 = nn.Linear(hidden_size, out_features)

    def forward(self, x):
        out = self.relu(self.fc1(x))
        out = self.fc2(out)
        return out


class DeepNeuralNet(nn.Module):
    def __init__(self, in_features, hidden_size, out_features):
        super(DeepNeuralNet, self).__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(in_features, hidden_size // 4)
        self.fc2 = nn.Linear(hidden_size // 4, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc5 = nn.Linear(hidden_size // 2, hidden_size // 4)
        self.fc6 = nn.Linear(hidden_size // 4, out_features)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.relu(out)
        out = self.fc4(out)
        out = self.relu(out)
        out = self.fc5(out)
        out = self.relu(out)
        out = self.fc6(out)
        return out
