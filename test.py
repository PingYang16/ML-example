# plot the learning curve
# ylabel from 800 to 4000
import matplotlib.pyplot as plt
import numpy as np
import pickle

import torch
from GCNmodel import GCN

from torch_geometric.data import DataLoader

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# load best parameters
with open('best_params.pkl', 'rb') as f:
    best_params = pickle.load(f)

# load best training and validation losses
best_train_loss = np.load('best_train_loss.npy')
best_val_loss = np.load('best_val_loss.npy')

figure1 = plt.figure()
figure1.plot(best_train_loss, label='train', color='red', alpha=0.5)
figure1.plot(best_val_loss, label='validation', color='blue', alpha=0.5)
figure1.xlabel('Epoch')
figure1.ylabel('Loss')
figure1.ylim(0, 6000)
figure1.legend()
figure1.savefig('learning_curve.png')

# load the best model
num_features = 5
num_classes = 1
param = best_params
batch_size = 32
model = GCN(num_features, num_classes, hidden_layers=param['hidden_layers'], hidden_size=param['hidden_size'])
model.load_state_dict(torch.load('best_model.pt'))

# load test data
test_data = torch.load('test_data.pt')

# define testing loader
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)
# predict on test set
model.eval()
preds = []
for data in test_loader:
    out = model(data)
    preds.append(out.detach().numpy())
preds = np.concatenate(preds)

# calculate R2 score
trues = np.concatenate([data.y.detach().numpy() for data in test_loader])
print(f"r2_score: {r2_score(trues, preds)}")

# calculate RMSE
print(f"RMSE: {np.sqrt(mean_squared_error(trues, preds))}")

# calculate MAE
print(f"MAE: {mean_absolute_error(trues, preds)}")

# plot a scatter plot of predicted vs. true values
# add a red line to show the perfect prediction
# generate second plot
figure2 = plt.figure()
figure2.scatter(trues, preds)
figure2.plot([-2250, -500], [-2250, -500], 'red')
figure2.xlabel('True values')
figure2.ylabel('Predicted values')

figure2.savefig('gcnmodel.png')