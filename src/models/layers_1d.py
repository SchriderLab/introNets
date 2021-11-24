# -*- coding: utf-8 -*-
import torch.nn as nn
import numpy as np
from collections import defaultdict

from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d as BN, Dropout
from networkx.algorithms.bipartite.matrix import from_biadjacency_matrix
from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

from torch import autograd
from typing import Callable, Any, Optional, Tuple, List

import warnings
from torch import Tensor

from torch_geometric.utils import to_dense_batch

from sparsenn.models.gcn.layers import DynamicGraphResBlock, GraphCyclicGRUBlock, GraphInceptionBlock
from torch_geometric.nn import global_mean_pool, MessageNorm
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
            
class GATCNet(nn.Module):
    def __init__(self):
        super(GATCNet, self).__init__()
        
        self.down = nn.ModuleList()
        self.up = nn.ModuleList()
        
        self.c1 = GraphInceptionBlock(255, 1)
        
        self.gcns_down = nn.ModuleList()
        self.gcns_up = nn.ModuleList()
        
        self.norms_down = nn.ModuleList()
        self.norms_up = nn.ModuleList()
        
        s = [127, 63, 31]
        filters = [2, 8, 8, 8]
        
        for ix in range(len(s)):
            self.down.append(nn.Conv2d(filters[ix], filters[ix + 1], (1, 3), 
                                       stride = (1, 2), padding = (0, 0), bias = False))
            self.norms_down.append(nn.BatchNorm2d(filters[ix + 1]))
            
            self.gcns_down.append(GraphInceptionBlock(s[ix], filters[ix + 1]))
            
        s = [255] + s
            
        for ix in range(len(s) - 1):
            self.up.append(nn.ConvTranspose2d(filters[-(ix + 1)], filters[-(ix + 2)], (1, 3), 
                                                            stride = (1, 2), padding = 0, bias = False))
            self.norms_up.append(nn.BatchNorm2d(filters[-(ix + 2)]))
            
            self.gcns_up.append(GraphInceptionBlock(s[-(ix + 2)], filters[-(ix + 2)] * 2))
            
            filters[-(ix + 2)] *= 2
            
        self.out = nn.Conv2d(8, 1, 1)
            
                        
    def forward(self, x, edge_index, batch, save_steps = False):
        x = to_dense_batch(x, batch)[0]
        x = torch.unsqueeze(x, dim = 1) # (B, 1, N, snps)
        
        x = torch.cat([x, self.c1(x, edge_index, batch)], dim = 1)
        
        if save_steps:
            steps = [x]
        
        xs = [x]
        for k in range(len(self.down)):
            # convolve
            x0 = self.norms_down[k](self.down[k](xs[-1]))
            
            # gcn
            x0 = self.gcns_down[k](x0, edge_index, batch) + x0
            
            xs.append(x0)
            
            if save_steps:
                steps.append(x0)

        x = xs[-1]        
        for k in range(len(self.up)):
            del xs[-1]            
            x = torch.cat([self.norms_up[k](self.up[k](x)), xs[-1]], dim = 1)

            # gcn
            x = self.gcns_up[k](x, edge_index, batch)
            
            if save_steps:
                steps.append(x)
        
        x = torch.cat([x[:,:,:150,:], x[:,:,150:,:]], dim = 1)
        x = torch.squeeze(self.out(x))
          
        return x

    
class Res1dBlock(nn.Module):
    def __init__(self, in_shape, out_channels, n_layers, 
                             k = 3, pooling = 2):
        super(Res1dBlock, self).__init__()
        
        in_shape = list(in_shape)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for ix in range(n_layers):
            self.convs.append(nn.Conv2d(in_shape[0], out_channels, (1, 3), stride = (1, 1), padding = (0, (k + 1) // 2 - 1)))
            self.norms.append(nn.LayerNorm(in_shape[1:]))
            
            in_shape[0] = out_channels
            
        self.pool = nn.MaxPool2d((1, 2), stride = (1, 2))
        
    def forward(self, x):
        xs = [self.norms[0](self.convs[0](x))]
        
        for ix in range(1, len(self.norms)):
            xs.append(self.norms[ix](self.convs[ix](xs[-1])) + xs[-1])
            
        x = torch.cat(xs, dim = 1).relu_()
        
        return self.pool(x)
    

class VanillaAttConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='add')  # "Add" aggregation (Step 5).
        self.norm = MessageNorm(True)

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]    

        # Step 3: Compute normalization.
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Step 4-5: Start propagating messages.
        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        # x_j has shape [E, out_channels]

        # Step 4: Normalize node features.
        return norm.view(-1, 1) * x_j
    
    def update(self, inputs, x):
        return self.norm(x, inputs)
    
class VanillaConv(MessagePassing):
    def __init__(self):
        super().__init__(aggr='add')  # "Add" aggregation (Step 5).
        self.norm = MessageNorm(True)

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]    

        # Step 3: Compute normalization.
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Step 4-5: Start propagating messages.
        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        # x_j has shape [E, out_channels]

        # Step 4: Normalize node features.
        return norm.view(-1, 1) * x_j
    
    def update(self, inputs, x):
        return self.norm(x, inputs)
        
class GATRelateCNet(nn.Module):
    def __init__(self, n_sites = 128, pop_size = 300, 
                         n_layers = 4, in_channels = 2, n_cycles = 1, hidden_channels = 16):
        super(GATRelateCNet, self).__init__()
        
        k_conv = 3
        pool_size = 2
        pool_stride = 2
        n_gat_layers = 3
        n_res_layers = 3
        
        res_channels = [64, 32, 16]
        up_channels = [16, 16, 16]
        
        self.down = nn.ModuleList()
        self.up = nn.ModuleList()
        
        self.gcns_down = nn.ModuleList()
        self.gcns_up = nn.ModuleList()
        
        self.norms_down = nn.ModuleList()
        self.norms_up = nn.ModuleList()
    
        ## stem
        self.stem_conv = nn.Sequential(nn.Conv2d(in_channels, 1, (1, k_conv), 
                                             stride = (1, 1), 
                                             padding = (0, 1), bias = True), 
                                             nn.LayerNorm((1, pop_size, n_sites)))
        
        self.gcn = VanillaConv()
        
        # after concatenating
        channels = 1
        stem_channels = copy.copy(channels)
        
        for ix in range(len(res_channels)):
            self.down.append(Res1dBlock((channels, pop_size, n_sites), res_channels[ix], n_res_layers))
            channels = res_channels[ix] * (n_res_layers)
            
            n_sites = n_sites // 2
                             
        print(res_channels)
        res_channels = list(np.array(res_channels)[::-1][1:]) + [128]
        print(res_channels)
        
        for ix in range(len(res_channels)):
            self.up.append(nn.ConvTranspose2d(channels, up_channels[ix], (1, 2), stride = (1, 2), padding = 0))
            self.norms_up.append(nn.BatchNorm2d(up_channels[ix]))
            channels = res_channels[ix] * n_res_layers + up_channels[ix]
        
        self.out = nn.Conv2d(17, 1, 1)
            
                        
    def forward(self, x, edge_index, batch, save_steps = False):
        #print('initial shape: {}'.format(x.shape))
        #print('edge_shape: {}'.format(edge_index.shape))
        batch_size, n_channels, n_ind, n_sites = x.shape
        
        x = self.stem_conv(x)
        
        n_channels = 1
        x = torch.flatten(x, 0, 2)
                
        #  insert graph convolution here...
        x = self.gcn(x, edge_index)
        ###################
        
        x = x.reshape(batch_size, n_channels, n_ind, n_sites)
        
        #print('after_stem: {}'.format(x.shape))
        
        xs = [x]
        for k in range(len(self.down)):
            xs.append(self.down[k](xs[-1]))
            #print('conv_down_{0}: {1}'.format(k, xs[-1].shape))
            
            n_sites = n_sites // 2
            n_channels = xs[-1].shape[1]
            xs[-1] = torch.flatten(xs[-1], 0, 2)   
            
            # insert graph convolution here...
            xs[-1] = self.gcn(xs[-1], edge_index)        
            ##################
            
            xs[-1] = xs[-1].reshape(batch_size, n_channels, n_ind, n_sites)

        x = xs[-1]        
        for k in range(len(self.up)):
            del xs[-1]         
            
            x = torch.cat([self.norms_up[k](self.up[k](x)), xs[-1]], dim = 1)
            print('conv_up_{0}: {1}'.format(k, x.shape))
            
            n_sites = n_sites * 2
            n_channels = x.shape[1]
            x = torch.flatten(x, 0, 2)  
            
            # insert graph convolution here...
            x = self.gcn(x, edge_index)
            ###################
            
            x = x.reshape(batch_size, n_channels, n_ind, n_sites)
                
        x = x[:,:,150:300,:]
        #print(x.shape)
        
        return torch.squeeze(self.out(x))
    

class GGRUCNet(nn.Module):
    def __init__(self, in_channels = 512, depth = 4):
        super(GGRUCNet, self).__init__()
        
        self.down = nn.ModuleList()
        self.up = nn.ModuleList()
        
        self.gcns_down = nn.ModuleList()
        self.gcns_up = nn.ModuleList()
        
        self.norms_down = nn.ModuleList()
        self.norms_up = nn.ModuleList()
        
        s = [449, 386, 323, 260, 197]
        
        for ix in range(len(s)):
            self.down.append(nn.Sequential(nn.Conv2d(1, 1, (1, 8), stride = (1, 2), padding = 0)))
            self.norms_down.append(nn.LayerNorm(s[ix]))
            
            self.gcns_down.append(GraphGRUBlock(s[ix], s[ix] // 2, 4, s[ix]))
            
        s = [512] + s
            
        for ix in range(len(s) - 1):
            self.up.append(nn.Sequential(nn.ConvTranspose2d(1, 1, (1, 8), stride = (1, 2), padding = 0)))
            self.norms_up.append(nn.LayerNorm(s[-(ix + 2)]))
            
            self.gcns_up.append(GraphGRUBlock(s[-(ix + 2)], s[-(ix + 2)] // 2, 4, s[-(ix + 2)]))
            
                        
    def forward(self, x, edge_index, batch):
        xs = [x]
        for k in range(len(self.down)):
            
            # convolve
            x0 = F.elu_(self.norms_down[k](self.down[k](xs[-1])))
            x0 = torch.squeeze(x0).view(batch.shape[0], x0.shape[-1])
            
            # gcn
            x0 = self.gcns_down[k](x0, edge_index)
            x0 = to_dense_batch(x0, batch)[0]
            x0 = torch.unsqueeze(x0, dim = 1)
        
            xs.append(x0)

        x = xs[-1]
        for k in range(len(self.up)):
            del xs[-1]
            
            x = F.elu_(self.norms_up[k](self.up[k](x))) + xs[-1]
            
            x = torch.squeeze(x).view(batch.shape[0], x.shape[-1])

            # gcn
            x = self.gcns_up[k](x, edge_index)
            
            x = to_dense_batch(x, batch)[0]
            x = torch.unsqueeze(x, dim = 1)
            
        
        x = torch.squeeze(x)
          
        return x
    
import matplotlib.pyplot as plt

class ConvTestModel(nn.Module):
    def __init__(self):
        super(ConvTestModel, self).__init__()
        
        self.down = nn.ModuleList()
        self.up = nn.ModuleList()

        s = [5, 5, 5, 5, 5]
        s_ = [1, 2, 1, 2, ]
        
        for ix in range(len(s)):
            self.down.append(nn.Sequential(nn.Conv2d(1, 1, (1, s[ix]), stride = (1, s_[ix]), padding = 0)))
            
        for ix in range(len(s)):
            self.up.append(nn.Sequential(nn.ConvTranspose2d(1, 1, (1, s[ix]), stride = (1, s_[ix]), padding = 0)))

            
                        
    def forward(self, x, edge_index, batch):
        x = to_dense_batch(x, batch)[0]
        x = torch.unsqueeze(x, dim = 1)
        
        print(x.shape)
        for k in range(len(self.down)):
            
            x = self.down[k](x)
            print(x.shape)
            
        for k in range(len(self.up)):
            x = self.up[k](x)
            print(x.shape)
            
        x = torch.squeeze(x)
          
        return x
    
if __name__ == '__main__':
    x = np.load('test_data/test_trans/000000.npz')
    
    model = GATRelateCNet(pop_size = 306)
    model.eval()
    
    print(model)
    
    edge_index = x['edge_index']
    x = x['x'][:,:128]
    x = np.stack([np.zeros((306, 128)), x])
    x = np.expand_dims(x, 0)
    
    print(x.shape)
    
    x = torch.FloatTensor(x)
    edge_index = torch.LongTensor(edge_index)
    batch = torch.LongTensor(np.zeros(x.shape[0]))
    
    x = model(x, edge_index, batch)
    
    print(x.shape)
    
    
    
    