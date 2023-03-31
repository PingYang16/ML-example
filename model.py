import scipy.io
import numpy as np

qm7 = scipy.io.loadmat('qm7.mat')
R = qm7['R']
Z = qm7['Z']
T = qm7['T'][0]

charge_to_atom = {
    1 : 'H',
    6 : 'C',
    7 : 'N',
    8 : 'O',
    16: 'S'
}

# unit Bohr to A
BohrToA = 0.529177249

for i in range(len(Z)):
    filename = f"xyz_file/qm7_xyz_{i+1}.xyz"
    atoms = []
    coordinates = []
    with open(filename, "w") as xyz:
        for j in range(len(Z[i])):
            if int(Z[i][j]) != 0:
                atoms.append(charge_to_atom[int(Z[i][j])])
                coordinates.append(R[i][j]*BohrToA)
        xyz.write(f"{len(atoms)}\n\n")
        for k in range(len(atoms)):
            xyz.write(f"{atoms[k]}  {coordinates[k][0]:.10f} {coordinates[k][1]:.10f} {coordinates[k][2]:.10f}\n")

import rdkit
from rdkit import Chem
from rdkit.Chem import rdmolfiles
from rdkit.Chem import AllChem

# convert each xyz to molecular graph
mols = []
for i in range(7165):
    filename = f"xyz_file/qm7_xyz_{i+1}.xyz"
    mol = rdmolfiles.MolFromXYZFile(filename)
    mols.append(mol)

# determine chemical bonds between each atom

from rdkit.Chem import rdDetermineBonds

for mol in mols: rdDetermineBonds.DetermineBonds(mol, charge=0)

# import required packages for graph convolutional network
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric import nn as gnn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

# build a 5-layer graph convolutional network (GCN) containing edge_index and edge_attr
# return the predicted energy of each molecule
class GCN(nn.Module):
    def __init__(self, num_features, num_classes):
        super(GCN, self).__init__()
        self.conv1 = gnn.GCNConv(num_features, 64)
        self.conv2 = gnn.GCNConv(64, 64)
        self.conv3 = gnn.GCNConv(64, 64)
        self.conv4 = gnn.GCNConv(64, 64)
        self.conv5 = gnn.GCNConv(64, 64)
        self.fc1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = F.relu(self.conv2(x, edge_index, edge_attr))
        x = F.relu(self.conv3(x, edge_index, edge_attr))
        x = F.relu(self.conv4(x, edge_index, edge_attr))
        x = F.relu(self.conv5(x, edge_index, edge_attr))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Calculate adjacency matrices for each molecule
adj_matrices = []
for mol in mols:
    adj_matrix = Chem.rdmolops.GetAdjacencyMatrix(mol)
    adj_matrices.append(adj_matrix)

# Make sure all adjacency matrices have the same dimension
max_atoms = max([adj.shape[0] for adj in adj_matrices])
for i in range(len(adj_matrices)):
    pad_width = max_atoms - adj_matrices[i].shape[0]
    adj_matrices[i] = np.pad(adj_matrices[i], pad_width=((0, pad_width), (0, pad_width)), mode='constant', constant_values=0)

# Calculate edge indices for each molecule
edge_indices = []
for adj in adj_matrices:
    edge_index = np.array(np.where(adj == 1)).T
    edge_indices.append(edge_index)

# Make sure all edge indices have the same dimension
max_edges = max([edge_index.shape[0] for edge_index in edge_indices])
for i in range(len(edge_indices)):
    pad_width = max_edges - edge_indices[i].shape[0]
    edge_indices[i] = np.pad(edge_indices[i], pad_width=((0, pad_width), (0, 0)), mode='constant', constant_values=0)

# Calculate edge attributes for each molecule
edge_attrs = []
for mol in mols:
    edge_attr = []
    for bond in mol.GetBonds():
        edge_attr.append(bond.GetBondTypeAsDouble())
    edge_attrs.append(edge_attr)

# Pad edge attributes corresponding to non-existing edges with zeros
for i in range(len(edge_attrs)):
    pad_width = max_edges - len(edge_attrs[i])
    edge_attrs[i] = np.pad(edge_attrs[i], pad_width=(0, pad_width), mode='constant', constant_values=0)

# Calculate node features for each molecule
node_features = []
for mol in mols:
    node_feature = []
    for atom in mol.GetAtoms():
        node_feature.append(atom.GetAtomicNum())
    node_features.append(node_feature)

# Make sure all node features have the same dimension (number of atoms, 1)
max_atoms = max([len(node_feature) for node_feature in node_features])
for i in range(len(node_features)):
    pad_width = max_atoms - len(node_features[i])
    node_features[i] = np.pad(node_features[i], pad_width=(0, pad_width), mode='constant', constant_values=0)

# Make a list of torch_geometric.data.Data objects
data_list = []
for i in range(len(mols)):
    data = Data(x=torch.tensor(node_features[i], dtype=torch.float).unsqueeze(1),
                edge_index=torch.tensor(edge_indices[i], dtype=torch.long).t().contiguous(),
                edge_attr=torch.tensor(edge_attrs[i], dtype=torch.float),
                y=torch.tensor(T[i], dtype=torch.float))
    data_list.append(data)

# Define training function
def train(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for data in loader:
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(loader.dataset)

# Define test function
def test(model, loader, criterion):
    model.eval()
    total_loss = 0
    for data in loader:
        out = model(data)
        loss = criterion(out, data.y)
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(loader.dataset)

# Split data into training set and test set
from sklearn.model_selection import train_test_split
train_data, test_data = train_test_split(data_list, test_size=0.2, random_state=42)

# define number of features and number of classes
num_features = 1
num_classes = 1
# define epochs
epochs = 100
# define learning rate
lr = 0.001
# define batch size
batch_size = 32

# define model
model = GCN(num_features, num_classes)
# define optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
# define loss function
criterion = nn.MSELoss()

# define training loader
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
# define test loader
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

# train model
for epoch in range(1, epochs + 1):
    train_loss = train(model, train_loader, optimizer, criterion)
    test_loss = test(model, test_loader, criterion)
    print(f'Epoch: {epoch:03d}, Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')

# predict on test set
model.eval()
preds = []
for data in test_loader:
    out = model(data)
    preds.append(out.detach().numpy())
preds = np.concatenate(preds)

# calculate R2 score
from sklearn.metrics import r2_score
r2_score(test_data.y.detach().numpy(), preds)

# calculate RMSE
from sklearn.metrics import mean_squared_error
np.sqrt(mean_squared_error(test_data.y.detach().numpy(), preds))

# print R2 score and RMSE
print(f'R2 score: {r2_score(test_data.y.detach().numpy(), preds):.4f}')