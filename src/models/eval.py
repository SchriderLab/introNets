# -*- coding: utf-8 -*-
from data_loaders import GCNDataGenerator
import numpy as np
import matplotlib.pyplot as plt

idir = '/mirror/introNets/ao_bf'
gen = GCNDataGenerator(idir, k = 16, batch_size = 4)
import torch

device = torch.device('cuda')

from layers import GCNUNet

model = GCNUNet(in_channels = 306, n_features = 306, n_classes = 1).to(device)
model.eval()

weights = '/mirror/introNets/training_results/ao_bf_regressor_r1/r1.weights'

checkpoint = torch.load(weights, map_location = device)
model.load_state_dict(checkpoint)

Y = []
Y_pred = []
for i in range(100):
    x, y, edges, batch = gen.get_batch()
    
    x = x.to(device)
    y = y.to(device)
    edges = [u.to(device) for u in edges]
    batch = batch.to(device)
    
    y_pred = model(x, edges, batch)
    
    y_pred = y_pred.detach().cpu().numpy().flatten()
    y = y.detach().cpu().numpy().flatten()
    
    Y.extend(y)
    Y_pred.extend(y_pred)
   
fig, axes = plt.subplots(ncols = 2)

axes[0].scatter(Y, Y_pred, alpha = 0.1)
axes[0].plot([0, np.max(Y)], [0, np.max(Y)], color = 'k')
axes[0].set_xlabel('mean introgression')
axes[0].set_ylabel('prediction')

axes[1].hist(np.array(Y_pred) - np.array(Y), bins = 35)


plt.savefig('eval_ao_bf.png', dpi = 100)
plt.close()