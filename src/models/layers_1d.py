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

from torch_geometric.nn.inits import glorot
            
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
    
class Res1dBlockUp(nn.Module):
    def __init__(self, in_shape, out_channels, n_layers, 
                             k = 3, pooling = 'max', up = False):
        super(Res1dBlockUp, self).__init__()
        
        in_shape = list(in_shape)
        
        # pass each pop through their own convolutions
        self.convs_l = nn.ModuleList()
        self.convs_r = nn.ModuleList()
        
        self.norms = nn.ModuleList()
        
        for ix in range(n_layers):
            self.convs_l.append(nn.Conv2d(in_shape[0], out_channels, (1, 3), 
                                        stride = (1, 1), padding = (0, (k + 1) // 2 - 1), bias = False))
            self.convs_r.append(nn.Conv2d(in_shape[0], out_channels, (1, 3), 
                                        stride = (1, 1), padding = (0, (k + 1) // 2 - 1), bias = False))
            
            self.norms.append(nn.Sequential(nn.InstanceNorm2d(out_channels), nn.Dropout(0.1)))
            
            in_shape[0] = out_channels
        
        self.up = nn.Upsample(scale_factor=(1,2), mode='bicubic', align_corners=True)
            
        self.activation = nn.ELU()
        
    def forward(self, x, edge_index, edge_attr, batch):
        x = self.up(x)
        
        batch_size, n_channels, n_ind, n_sites = x.shape
        
        xs = []
        
        xl = self.convs_l[0](x[:,:,:n_ind // 2,:])
        xr = self.convs_r[0](x[:,:,n_ind // 2:,:])
        
        xs.append(torch.cat([xl, xr], dim = 2))
        
        for ix in range(1, len(self.norms)):
            xl = self.convs_l[ix](xs[-1][:,:,:n_ind // 2,:])
            xr = self.convs_r[ix](xs[-1][:,:,n_ind // 2:,:])
            
            xs.append(torch.cat([xl, xr], dim = 2))
        
            xs[-1] = self.norms[ix](xs[-1] + xs[-2])
            
        x = torch.cat([x, self.activation(torch.cat(xs, dim = 1))], dim = 1)
        
        return x
    
class Res1dGraphBlockUp(nn.Module):
    def __init__(self, in_shape, out_channels, n_layers, 
                             k = 3, pooling = 'max', up = False):
        super(Res1dGraphBlockUp, self).__init__()
        
        in_shape = list(in_shape)
        
        # pass each pop through their own convolutions
        self.convs_l = nn.ModuleList()
        self.convs_r = nn.ModuleList()
        
        self.norms = nn.ModuleList()
        self.gcns = nn.ModuleList()
        
        for ix in range(n_layers):
            self.convs_l.append(nn.Conv2d(in_shape[0], out_channels, (1, 3), 
                                        stride = (1, 1), padding = (0, (k + 1) // 2 - 1), bias = False))
            self.convs_r.append(nn.Conv2d(in_shape[0], out_channels, (1, 3), 
                                        stride = (1, 1), padding = (0, (k + 1) // 2 - 1), bias = False))
            
            self.gcns.append(VanillaAttConv())
            self.norms.append(nn.Sequential(nn.InstanceNorm2d(out_channels), nn.Dropout(0.1)))
            
            in_shape[0] = out_channels
        
        self.up = nn.Upsample(scale_factor=(1,2), mode='bicubic', align_corners=True)
            
        self.activation = nn.ELU()
        
    def forward(self, x, edge_index, edge_attr, batch):
        x = self.up(x)
        
        batch_size, n_channels, n_ind, n_sites = x.shape
        
        xs = []
        
        xl = self.convs_l[0](x[:,:,:n_ind // 2,:])
        xr = self.convs_r[0](x[:,:,n_ind // 2:,:])
        
        xs.append(torch.cat([xl, xr], dim = 2))
        
        n_channels = xs[-1].shape[1]
        xs[-1] = torch.flatten(xs[-1].transpose(1, 2), 2, 3).flatten(0, 1)   
        
        # insert graph convolution here...
        xs[-1] = self.gcns[0](xs[-1], edge_index, edge_attr)     
        ##################
        
        xs[-1] = to_dense_batch(xs[-1], batch)[0]
        xs[-1] = xs[-1].reshape(batch_size, n_ind, n_channels, n_sites).transpose(1, 2)
        xs[-1] = self.norms[0](xs[-1])
        
        for ix in range(1, len(self.norms)):
            xl = self.convs_l[ix](xs[-1][:,:,:n_ind // 2,:])
            xr = self.convs_r[ix](xs[-1][:,:,n_ind // 2:,:])
            
            xs.append(torch.cat([xl, xr], dim = 2))
        
            n_channels = xs[-1].shape[1]
            xs[-1] = torch.flatten(xs[-1].transpose(1, 2), 2, 3).flatten(0, 1)   
            
            # insert graph convolution here...
            xs[-1] = self.gcns[ix](xs[-1], edge_index, edge_attr)     
            ##################
            
            xs[-1] = to_dense_batch(xs[-1], batch)[0]
            xs[-1] = xs[-1].reshape(batch_size, n_ind, n_channels, n_sites).transpose(1, 2)
            xs[-1] = self.norms[ix](xs[-1] + xs[-2])
            
        x = torch.cat([x, self.activation(torch.cat(xs, dim = 1))], dim = 1)
        
        return x

class Res1dGraphBlock(nn.Module):
    def __init__(self, in_shape, out_channels, n_layers, gcn_channels = 4,
                             k = 3, pooling = 'max', up = False):
        super(Res1dGraphBlock, self).__init__()
        
        in_shape = list(in_shape)
        
        # pass each pop through their own convolutions
        self.convs_l = nn.ModuleList()
        self.convs_r = nn.ModuleList()
        
        self.gcn_convs = nn.ModuleList()
        
        self.norms = nn.ModuleList()
        self.gcns = nn.ModuleList()
        
        for ix in range(n_layers):
            self.convs_l.append(nn.Conv2d(in_shape[0], out_channels, (1, 3), 
                                        stride = (1, 1), padding = (0, (k + 1) // 2 - 1), bias = False))
            self.convs_r.append(nn.Conv2d(in_shape[0], out_channels, (1, 3), 
                                        stride = (1, 1), padding = (0, (k + 1) // 2 - 1), bias = False))
            
            # for down sampling the dimensionality of the features for the gcn part
            self.gcn_convs.append(nn.Conv2d(out_channels, gcn_channels, 1, 1))
            
            self.gcns.append(VanillaAttConv())
            self.norms.append(nn.Sequential(nn.InstanceNorm2d(out_channels), nn.Dropout(0.1)))
            
            in_shape[0] = out_channels
        
        if pooling == 'max':
            self.pool = nn.MaxPool2d((1, 2), stride = (1, 2))
        else:
            self.pool = None
            
        self.activation = nn.ELU()
        
    def forward(self, x, edge_index, edge_attr, batch):
        batch_size, n_channels, n_ind, n_sites = x.shape
        
        xs = []
        xgs = []
        
        xl = self.convs_l[0](x[:,:,:n_ind // 2,:])
        xr = self.convs_r[0](x[:,:,n_ind // 2:,:])
        
        xs.append(torch.cat([xl, xr], dim = 2))
        
        # the graph features at this point in the network
        xg = self.gcn_convs[0](xs[-1])
        n_channels = xg.shape[1]

        xg = torch.flatten(xg.transpose(1, 2), 2, 3).flatten(0, 1)   

        # insert graph convolution here...
        xg = self.gcns[0](xg, edge_index, edge_attr)

        ##################
        
        xg = to_dense_batch(xg, batch)[0]
        xg = xg.reshape(batch_size, n_ind, n_channels, n_sites).transpose(1, 2)
        
        # this will have out_channels + graph channels
        xs[-1] = self.norms[0](torch.cat([xg, xs[-1]]))
        
        for ix in range(1, len(self.norms)):
            xl = self.convs_l[ix](xs[-1][:,:,:n_ind // 2,:])
            xr = self.convs_r[ix](xs[-1][:,:,n_ind // 2:,:])
            
            xs.append(torch.cat([xl, xr], dim = 2))

            xg = self.gcn_convs[ix](xs[-1])
            print(xg.shape, xs[-1].shape)
            
            n_channels = xg.shape[1]
    
            xg = torch.flatten(xg.transpose(1, 2), 2, 3).flatten(0, 1)   
            
            # insert graph convolution here...
            xg = self.gcns[ix](xg, edge_index, edge_attr)     
            ##################
            
            xg = to_dense_batch(xg, batch)[0]
            xg = xg.reshape(batch_size, n_ind, n_channels, n_sites).transpose(1, 2)
            
            # this will have out_channels + graph channels
            # concatenate the graph features and add the previous for a residual connection
            xs[-1] = self.norms[ix](torch.cat([xg, xs[-1]], dim = 1) + xs[-2])
                
        x = self.activation(torch.cat(xs, dim = 1))
        
        if self.pool is not None:
            return self.pool(x)
        else:
            return x

class VanillaAttConv(MessagePassing):
    def __init__(self, negative_slope = 0.2, leaky = True):
        super().__init__(aggr='add')  # "Add" aggregation (Step 5).
        self.norm = MessageNorm(True)
        self.negative_slope = negative_slope
        
        _ = [nn.BatchNorm1d(8), nn.Linear(8, 128), nn.LayerNorm((128,)), nn.ELU(), 
                                     nn.Linear(128, 128), nn.LayerNorm((128,)), nn.ELU(), nn.Linear(128, 1)]
        
        if leaky:
            _.append(nn.LeakyReLU(negative_slope = negative_slope))
            _.append(nn.Sigmoid())
        
        self.att_mlp = nn.Sequential(*_)
        
    
    def forward(self, x, edge_index, edge_attr):
        att = self.att_mlp(edge_attr)
        
        return self.propagate(edge_index, x = x, att = att)

    def message(self, x_j, att):
        return x_j * att
    
    def update(self, inputs, x):
        return self.norm(x, inputs)
    
class VanillaConv(MessagePassing):
    def __init__(self, negative_slope = 0.2):
        super().__init__(aggr='add')  # "Add" aggregation (Step 5).
        self.norm = MessageNorm(True)
        self.negative_slope = negative_slope

    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x)

    def message(self, x_j):
        return x_j
    
    def update(self, inputs, x):
        return self.norm(x, inputs)
    
class Res1dBlock(nn.Module):
    def __init__(self, in_shape, out_channels, n_layers, 
                             k = 3, pooling = 'max'):
        super(Res1dBlock, self).__init__()
        
        in_shape = list(in_shape)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for ix in range(n_layers):
            self.convs.append(nn.Conv2d(in_shape[0], out_channels, (1, 3), 
                                        stride = (1, 1), padding = (0, (k + 1) // 2 - 1)))
            self.norms.append(nn.LayerNorm(in_shape[1:]))
            
            in_shape[0] = out_channels
        
        if pooling == 'max':
            self.pool = nn.MaxPool2d((1, 2), stride = (1, 2))
        else:
            self.pool = None
            
        self.activation = nn.ELU()
        
    def forward(self, x):
        xs = [self.norms[0](self.convs[0](x))]
        
        for ix in range(1, len(self.norms)):
            xs.append(self.norms[ix](self.convs[ix](xs[-1])) + xs[-1])
            
        x = self.activation(torch.cat(xs, dim = 1))
        
        if self.pool is not None:
            return self.pool(x)
        else:
            return x
        
### Notes:
## As it stands, this is a traditional UNet with (ResBlock + MaxPool) as the down sampling routine
## and ConvTranspose2D as the upsample
## -----
## There is a "stem" that takes the (genotype matrix (cat) breakpoints) -> 1 channel image.
## After the stem and each convolution operation (up and down) there is a GCN.

## i1 (had a single sum operator across edges)
## i2 (introduced edge attrs) + edge conditioned single sigmoid function to
##### go from edge attr -> a; giving x_j = x_j * a
    
class GATRelateCNet(nn.Module):
    def __init__(self, n_sites = 128, pop_size = 300, pred_pop = 1,
                         n_layers = 4, in_channels = 2, 
                         n_cycles = 1, hidden_channels = 16):
        super(GATRelateCNet, self).__init__()
        
        k_conv = 3
        pool_size = 2
        pool_stride = 2
        n_gat_layers = 3
        n_res_layers = 3
        
        stem_channels = 2
        
        res_channels = [32, 16, 8]
        up_channels = [16, 16, 16]
        
        self.pred_pop = pred_pop
        
        # Two sets of convolutional filters
        self.down_l = nn.ModuleList()
        self.up_l = nn.ModuleList()
        
        self.down_r = nn.ModuleList()
        self.up_r = nn.ModuleList()
        
        self.norms_down = nn.ModuleList()
        self.norms_up = nn.ModuleList()
        
        self.gcns_down = nn.ModuleList()
        self.gcns_up = nn.ModuleList()
    
        ## stem
        self.stem_conv = nn.Sequential(nn.Conv2d(in_channels, stem_channels, (1, k_conv), 
                                             stride = (1, 1), 
                                             padding = (0, 1), bias = True))
        
        self.stem_gcn = VanillaAttConv()
        self.stem_norm = nn.LayerNorm((stem_channels, pop_size, n_sites))
        
        # after concatenating
        channels = stem_channels + in_channels
        
        for ix in range(len(res_channels)):
            
            self.down_l.append(Res1dBlock((channels, pop_size // 2, n_sites), res_channels[ix], n_res_layers))
            self.down_r.append(Res1dBlock((channels, pop_size // 2, n_sites), res_channels[ix], n_res_layers))
            
            self.norms_down.append(nn.InstanceNorm2d(res_channels[ix] * (n_res_layers)))
            self.gcns_down.append(VanillaAttConv())
            
            channels = res_channels[ix] * (n_res_layers)
            
            n_sites = n_sites // 2
                             
        print(res_channels)
        res_channels = list(np.array(res_channels)[::-1][1:]) + [128]
        print(res_channels)
        
        for ix in range(len(res_channels)):
            n_sites *= 2
            
            self.up_l.append(nn.ConvTranspose2d(channels, up_channels[ix], (1, 2), stride = (1, 2), padding = 0))
            self.up_r.append(nn.ConvTranspose2d(channels, up_channels[ix], (1, 2), stride = (1, 2), padding = 0))
            
            self.norms_up.append(nn.InstanceNorm2d(up_channels[ix]))
            self.gcns_up.append(VanillaAttConv())
            
            channels = res_channels[ix] * n_res_layers + up_channels[ix]
    
        self.final_conv = nn.Conv2d(up_channels[-1] + (in_channels + stem_channels), 1, 1)
                        
    def forward(self, x, edge_index, edge_attr, batch, save_steps = False):
        #print('initial shape: {}'.format(x.shape))
        #print('edge_shape: {}'.format(edge_index.shape))
        batch_size, n_channels, n_ind, n_sites = x.shape
        
        x0 = self.stem_conv(x)
        
        n_channels = x0.shape[1]
        x0 = torch.flatten(x0.transpose(1, 2), 2, 3).flatten(0, 1)
                
        #  insert graph convolution here...
        x0 = self.stem_gcn(x0, edge_index, edge_attr)
        ###################
        
        x0 = to_dense_batch(x0, batch)[0]
        x0 = x0.reshape(batch_size, n_ind, n_channels, n_sites).transpose(1, 2)
        
        x0 = self.stem_norm(x0).relu_()      
        x0 = torch.cat([x, x0], dim = 1)
    
        #print('after_stem: {}'.format(x.shape))
        
        xs = [x0]
        for k in range(len(self.down_l)):
            # pass each pop to it's 1d conv
            xl = self.down_l[k](xs[-1][:,:,:n_ind // 2,:])
            xr = self.down_r[k](xs[-1][:,:,n_ind // 2:,:])
            
            xs.append(torch.cat([xl, xr], dim = 2))
            #print('conv_down_{0}: {1}'.format(k, xs[-1].shape))
            
            n_sites = n_sites // 2
            n_channels = xs[-1].shape[1]
            xs[-1] = torch.flatten(xs[-1].transpose(1, 2), 2, 3).flatten(0, 1)   
            
            # insert graph convolution here...
            xs[-1] = self.gcns_down[k](xs[-1], edge_index, edge_attr)     
            ##################
            
            xs[-1] = to_dense_batch(xs[-1], batch)[0]
            xs[-1] = xs[-1].reshape(batch_size, n_ind, n_channels, n_sites).transpose(1, 2)
            
            xs[-1] = self.norms_down[k](xs[-1]).relu_()

        # x0 has the original catted with it so overwrite x
        x = xs[-1]        
        for k in range(len(self.up_l)):
            del xs[-1]
            
            # pass each pop to it's 1d conv
            xl = self.up_l[k](x[:,:,:n_ind // 2,:])
            xr = self.up_r[k](x[:,:,n_ind // 2:,:])
            
            x = torch.cat([xl, xr], dim = 2)
            
            #print('conv_up_{0}: {1}'.format(k, x.shape))
            
            n_sites = xs[-1].shape[-1]
            n_channels = x.shape[1]
            x = torch.flatten(x.transpose(1, 2), 2, 3).flatten(0, 1)   
            
            # insert graph convolution here...
            x = self.gcns_up[k](x, edge_index, edge_attr)
            ###################
            
            x = to_dense_batch(x, batch)[0]
            x = x.reshape(batch_size, n_ind, n_channels, n_sites).transpose(1, 2)
            
            x = self.norms_up[k](x).relu_()
            
            x = torch.cat([x, xs[-1]], dim = 1)
        
        # gc
        del x0
        del xl
        del xr
        
        # we only want the second pop
        if self.pred_pop == 1:
            x = x[:,:,n_ind // 2:,:]
        # we only want the first pop
        elif self.pred_pop == 0:
            x = x[:,:,:n_ind // 2,:]
        
        # go back to one channel
        x = torch.squeeze(self.final_conv(x))

        return x
    
## Notes:
## Making a copy of the class as V2 because I feel this is a big enough design change, putting gcns at every 1d convolution
## Made the class Res1dGraphBlock.  This makes the definition of the net much simpler / consolidated.

##
class GATRelateCNetV2(nn.Module):
    def __init__(self, n_sites = 128, pop_size = 300, pred_pop = 1,
                         n_layers = 4, in_channels = 2, 
                         n_cycles = 1, hidden_channels = 16, 
                         graph_up = False):
        super(GATRelateCNetV2, self).__init__()
        
        k_conv = 3
        pool_size = 2
        pool_stride = 2
        n_gat_layers = 3
        n_res_layers = 3
        
        stem_channels = 2
        graph_channels = 8
        
        res_channels = [8, 16, 32]
        up_channels = [32, 16, 8]
        
        self.pred_pop = pred_pop
        
        # Two sets of convolutional filters
        self.down = nn.ModuleList()
        self.up = nn.ModuleList()
        
        self.norms_up = nn.ModuleList()
        self.norms_down = nn.ModuleList()
    
        ## stem
        self.stem_conv = nn.Sequential(nn.Conv2d(in_channels, stem_channels, (1, k_conv), 
                                             stride = (1, 1), 
                                             padding = (0, 1), bias = True))
        
        self.stem_gcn = VanillaAttConv()
        self.stem_norm = nn.InstanceNorm2d(stem_channels)
        
        # after concatenating
        channels = stem_channels + in_channels
        
        for ix in range(len(res_channels)):
            self.down.append(Res1dGraphBlock((channels, pop_size, n_sites), res_channels[ix], n_res_layers, gcn_channels = graph_channels))
            self.norms_down.append(nn.Dropout(0.1))
            channels = res_channels[ix] * (n_res_layers)
            
            n_sites = n_sites // 2
                             
        print(res_channels)
        res_channels = list(np.array(res_channels)[::-1][1:]) + [128]
        print(res_channels)
        
        for ix in range(len(res_channels)):
            n_sites *= 2
            
            if not graph_up:
                self.up.append(Res1dBlockUp((channels, pop_size, n_sites), up_channels[ix] // 2, 2))
            else:
                self.up.append(Res1dGraphBlockUp((channels, pop_size, n_sites), up_channels[ix] // 2, 2))
            self.norms_up.append(nn.InstanceNorm2d(up_channels[ix]))
            
            if ix != len(res_channels) - 1:
                channels = res_channels[ix] * (n_res_layers) + channels + up_channels[ix]
    
        self.final_conv = nn.Conv2d(up_channels[-1] + channels + (in_channels + stem_channels), 1, 1)
        self.act = nn.ELU()
                        
    def forward(self, x, edge_index, edge_attr, batch, save_steps = False):
        #print('initial shape: {}'.format(x.shape))
        #print('edge_shape: {}'.format(edge_index.shape))
        batch_size, n_channels, n_ind, n_sites = x.shape
        
        x0 = self.stem_conv(x)
        
        n_channels = x0.shape[1]
        x0 = torch.flatten(x0.transpose(1, 2), 2, 3).flatten(0, 1)
                
        #  insert graph convolution here...
        x0 = self.stem_gcn(x0, edge_index, edge_attr)
        ###################
        
        x0 = to_dense_batch(x0, batch)[0]
        x0 = x0.reshape(batch_size, n_ind, n_channels, n_sites).transpose(1, 2)
        
        x0 = self.act(self.stem_norm(x0))     
        x0 = torch.cat([x, x0], dim = 1)
    
        #print('after_stem: {}'.format(x.shape))
        
        xs = [x0]
        for k in range(len(self.down)):
            # pass each pop to it's 1d conv
            xs.append(self.norms_down[k](self.down[k](xs[-1], edge_index, edge_attr, batch)))

        # x0 has the original catted with it so overwrite x
        x = xs[-1]        
        for k in range(len(self.up)):
            del xs[-1]
            
            x = self.norms_up[k](self.up[k](x, edge_index, edge_attr, batch))
            x = torch.cat([x, xs[-1]], dim = 1)
        
        # gc
        del x0
        
        # we only want the second pop
        if self.pred_pop == 1:
            x = x[:,:,n_ind // 2:,:]
        # we only want the first pop
        elif self.pred_pop == 0:
            x = x[:,:,:n_ind // 2,:]
        
        # go back to one channel
        x = torch.squeeze(self.final_conv(x))

        return x


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
    
    
    
    