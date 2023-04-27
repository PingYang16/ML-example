import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric import nn as gnn

# build a graph convolutional network (GCN) containing edge_index and edge_attr
# return the predicted energy of each molecule
# hidden layer size and number of hidden layers are adjustable
class GCN(nn.Module):
    def __init__(self, num_features, num_classes, hidden_layers=5, hidden_size=64):
        super(GCN, self).__init__()
        self.conv1 = gnn.GCNConv(num_features, hidden_size)
        self.hidden_layers = nn.ModuleList([gnn.GCNConv(hidden_size, hidden_size) for _ in range(hidden_layers-1)])
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        for layer in self.hidden_layers:
            x = F.relu(layer(x, edge_index, edge_attr))
        x = gnn.global_mean_pool(x, data.batch)
        x = self.fc(x)
        return x
    
def train(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for data in loader:
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data.y.view(-1, 1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(loader.dataset)

def validate(model, loader, criterion):
    model.eval()
    total_loss = 0
    for data in loader:
        out = model(data)
        loss = criterion(out, data.y.view(-1, 1))
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(loader.dataset)