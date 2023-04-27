import os
import sys
sys.path.append(os.getcwd())

import scipy.io

import rdkit
from rdkit import Chem
from rdkit.Chem import rdmolfiles
from rdkit.Chem import rdDetermineBonds

import numpy as np

import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split

from torch_geometric import nn as gnn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from sklearn.model_selection import ParameterGrid

from GCNmodel import GCN, train, validate

qm7 = scipy.io.loadmat('qm7.mat')
R = qm7['R']
Z = qm7['Z']
T = qm7['T'][0]

# hybridization state one-hot encoding
HybridizationToFeature = {
    rdkit.Chem.rdchem.HybridizationType.SP3 : 4,
    rdkit.Chem.rdchem.HybridizationType.SP2 : 3,
    rdkit.Chem.rdchem.HybridizationType.SP  : 2,
    rdkit.Chem.rdchem.HybridizationType.S   : 1
}

mols = []
for i in range(7165):
    filename = f"xyz_file/qm7_xyz_{i+1}.xyz"
    mol = rdmolfiles.MolFromXYZFile(filename)
    mols.append(mol)

for mol in mols: rdDetermineBonds.DetermineBonds(mol, charge=0)

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

# Calculate edge attributes for each molecule
edge_attrs = []
for mol in mols:
    edge_attr = []
    for bond in mol.GetBonds():
        edge_attr.append(bond.GetBondTypeAsDouble())
    edge_attrs.append(edge_attr)

# Make sure all edge indices have the same dimension
max_edges = max([edge_index.shape[0] for edge_index in edge_indices])
for i in range(len(edge_indices)):
    pad_width = max_edges - edge_indices[i].shape[0]
    edge_indices[i] = np.pad(edge_indices[i], pad_width=((0, pad_width), (0, 0)), mode='constant', constant_values=0)

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

# Add atomic coordinates R matrix as node features
for i in range(len(mols)):
    coordinates = np.transpose(R[i])
    node_features[i] = np.concatenate(([node_features[i]], coordinates), axis=0)

# Add another node feature as hybridization of each atom
for i in range(len(mols)):
    hybridization = []
    for atom in mols[i].GetAtoms():
        hybridization.append(HybridizationToFeature[atom.GetHybridization()])
    while len(hybridization) < max_atoms:
        hybridization.append(0)
    # add zeros to the end of the list to make sure the length of the list is the same as the number of atoms
    node_features[i] = np.concatenate((node_features[i], [hybridization]), axis=0)

# Make a list of torch_geometric.data.Data objects
data_list = []
for i in range(len(mols)):
    data = Data(x=torch.tensor(np.transpose(node_features[i]), dtype=torch.float), 
                edge_index=torch.tensor(edge_indices[i], dtype=torch.long).T, 
                edge_attr=torch.tensor(edge_attrs[i], dtype=torch.float).unsqueeze(1), 
                y=torch.tensor(T[i], dtype=torch.float)
)
    data_list.append(data)

torch.manual_seed(12345)
train_data, val_data, test_data = random_split(data_list, [0.7, 0.15, 0.15])

# define hyperparameters to search
params = {
    'hidden_size': [128, 256, 512],
    'hidden_layers': [4, 6, 8],
    'lr': [0.001, 0.01, 0.1]
}
# create a grid of hyperparameters
grid = ParameterGrid(params)

# define number of features and number of classes
num_features = 5
num_classes = 1
# define epochs
epochs = 1500
batch_size = 32

best_val = 2000
best_params = None
# iterate over hyperparameter combinations
for param in grid:
    # define model
    model = GCN(num_features, num_classes, hidden_layers=param['hidden_layers'], hidden_size=param['hidden_size'])
    # define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=param['lr'])
    # define loss function
    criterion = nn.MSELoss()

    # define training loader
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    # define validating loader
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    train_loss_data = []
    val_loss_data = []
    # train model
    for epoch in range(1, epochs + 1):
        train_loss = train(model, train_loader, optimizer, criterion)
        val_loss = validate(model, val_loader, criterion)
        train_loss_data.append(train_loss)
        val_loss_data.append(val_loss)
        if val_loss < best_val:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pt')
            # store the best parameters
            best_params = param
            # store the best training and validation losses
            best_train_loss = train_loss_data
            best_val_loss = val_loss_data
        if epoch % 100 == 0:
            print('Epoch: {:03d}, Train Loss: {:.7f}, Val Loss: {:.7f}'.format(epoch, train_loss, val_loss))

# print and save the best parameters
print('Best parameters: ', best_params)
with open('best_params.pkl', 'wb') as f:
    pickle.dump(best_params, f)
# save the best training and validation losses
np.save('best_train_loss.npy', best_train_loss)
np.save('best_val_loss.npy', best_val_loss)

# save the test data
torch.save(test_data, 'test_data.pt')