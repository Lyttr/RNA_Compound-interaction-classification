
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=1024):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(0.1)

        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(0.1)

        self.fc3 = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        #x = self.drop1(x)

        x = self.fc2(x)
        x = self.relu2(x)
        #x = self.drop2(x)

        x = self.fc3(x)
        x = self.sigmoid(x)
        return x.squeeze(1)
