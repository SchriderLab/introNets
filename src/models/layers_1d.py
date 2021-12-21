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

import scipy.signal
import scipy.optimize
import torch
from torch_utils import misc
from torch_utils import persistence
from torch_utils.ops import conv2d_gradfix
from torch_utils.ops import filtered_lrelu
from torch_utils.ops import bias_act

from torch import autograd
from typing import Callable, Any, Optional, Tuple, List

import warnings
from torch import Tensor

from torch_geometric.utils import to_dense_batch
from torch_scatter import scatter_max, scatter, scatter_mean, scatter_std

from sparsenn.models.gcn.layers import DynamicGraphResBlock, GraphCyclicGRUBlock, GraphInceptionBlock
from torch_geometric.nn import global_mean_pool, MessageNorm
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

from torch_geometric.nn.inits import glorot
from torch_utils.ops import filtered_lrelu
from torch_geometric.nn import LayerNorm
            
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
            
            self.norms.append(nn.Sequential(nn.InstanceNorm2d(out_channels), nn.Dropout2d(0.1)))
            
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
                             k = 3, pooling = 'max', up = False, att_activation = 'sigmoid'):
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
            
            self.gcns.append(VanillaAttConv(activation = att_activation))
            self.norms.append(nn.Sequential(nn.InstanceNorm2d(out_channels), nn.Dropout2d(0.1)))
            
            in_shape[0] = out_channels + gcn_channels
        
        if pooling == 'max':
            self.pool = nn.MaxPool2d((1, 2), stride = (1, 2))
        else:
            self.pool = None
            
        self.activation = nn.ReLU()
        
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
        xs[-1] = self.norms[0](torch.cat([xg, xs[-1]], dim = 1))
        
        for ix in range(1, len(self.norms)):
            xl = self.convs_l[ix](xs[-1][:,:,:n_ind // 2,:])
            xr = self.convs_r[ix](xs[-1][:,:,n_ind // 2:,:])
            
            xs.append(torch.cat([xl, xr], dim = 2))

            xg = self.gcn_convs[ix](xs[-1])
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
    def __init__(self, negative_slope = 0.2, activation = 'sigmoid'):
        super().__init__(aggr='add')  # "Add" aggregation (Step 5).
        self.norm = MessageNorm(True)
        self.negative_slope = negative_slope
        
        _ = [nn.BatchNorm1d(8), nn.Linear(8, 16), nn.LayerNorm((16,)), nn.ReLU(), 
                                     nn.Linear(16, 32), nn.LayerNorm((32,)), nn.ReLU(), nn.Linear(32, 1)]
        
        if activation == 'sigmoid':
            _.append(nn.LeakyReLU(negative_slope = negative_slope))
            _.append(nn.Sigmoid())
        elif activation == 'tanh':
            _.append(nn.LeakyReLU(negative_slope = negative_slope))
            _.append(nn.Tanh())
        
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
            self.norms.append(nn.Sequential(nn.InstanceNorm2d(out_channels), nn.Dropout2d(0.1)))
            
            in_shape[0] = out_channels
        
        if pooling == 'max':
            self.pool = nn.MaxPool2d((1, 2), stride = (1, 2))
        else:
            self.pool = None
            
        self.activation = nn.ELU()
        
    def forward(self, x, return_unpooled = False):
        xs = [self.norms[0](self.convs[0](x))]
        
        for ix in range(1, len(self.norms)):
            xs.append(self.norms[ix](self.convs[ix](xs[-1])) + xs[-1])
            
        x = self.activation(torch.cat(xs, dim = 1))
        
        if self.pool is not None:
            xp = self.pool(x)
        else:
            xp = x
            
        if return_unpooled:
            return xp, x
        else:
            return xp
        
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
        
        res_channels = [64, 128, 256]
        up_channels = [256, 128, 64]
        
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
    
        # we're going to concatenate global features to the pred pop and do one more residual 1d conv
        self.xl_final_down = nn.Conv2d(up_channels[-1] + (in_channels + stem_channels), 32, 1, 1)
        self.xr_final_down = nn.Conv2d(up_channels[-1] + (in_channels + stem_channels), 32, 1, 1)
        
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
        x = torch.cat([x, x0], dim = 1)
    
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
        
        xl = self.xl_final_down(x[:,:,:n_ind // 2,:])
        xl = torch.flatten(xl.transpose(1, 2), 2, 3).flatten(0, 1)
        xl = torch.squeeze(self.activation(xl))
        
        xr = self.xr_final_down(x[:,:,n_ind // 2:,:])
        xr = torch.flatten(xr.transpose(1, 2), 2, 3).flatten(0, 1)
        xr = torch.squeeze(self.activation(xr))
        
        xr = to_dense_batch(xr, batch)[0]
        xr = xr.reshape(batch_size, n_ind, n_channels, n_sites).transpose(1, 2)
        
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
                         n_layers = 4, in_channels = 1, 
                         n_cycles = 1, hidden_channels = 16, 
                         graph_up = False, att_activation = 'sigmoid'):
        super(GATRelateCNetV2, self).__init__()
        
        k_conv = 3
        pool_size = 2
        pool_stride = 2
        n_gat_layers = 3
        n_res_layers = 3
        
        stem_channels = 2
        graph_channels = 8
        
        res_channels = [16, 32, 64]
        up_channels = [64, 32, 16]
        
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
        
        self.stem_gcn = VanillaAttConv(activation = att_activation)
        self.stem_norm = nn.InstanceNorm2d(stem_channels)
        
        # after concatenating
        channels = stem_channels + in_channels
        
        for ix in range(len(res_channels)):
            self.down.append(Res1dGraphBlock((channels, pop_size, n_sites), res_channels[ix], n_res_layers, 
                                             gcn_channels = graph_channels, att_activation = att_activation))
            self.norms_down.append(nn.Dropout(0.1))
            channels = res_channels[ix] * (n_res_layers) + graph_channels * n_res_layers
            
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
                channels = res_channels[ix] * (n_res_layers) + graph_channels * n_res_layers + channels + up_channels[ix]
    
        # we're going to concatenate global features to the pred pop and do one more residual 1d conv
        #self.xl_final_down = nn.Conv2d(up_channels[-1] + channels + (in_channels + stem_channels), 4, 1, 1)
        #self.xr_final_down = nn.Conv2d(up_channels[-1] + channels + (in_channels + stem_channels), 4, 1, 1)
        
        self.out_channels = up_channels[-1] + channels + (in_channels + stem_channels)
        self.final_down = nn.Sequential(nn.Conv2d(self.out_channels, 128, 1, 1), 
                                        Res1dBlock((128, pop_size, n_sites), 64, 2, pooling = None))
    
        self.final_conv = nn.Conv2d(128, 1, 1)
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
        
        # downsample through double conv
        x = self.final_down(x)
        
        # go back to one channel
        x = torch.squeeze(self.final_conv(x))

        return x
    
from torch.nn import Parameter
from torch_sparse import SparseTensor, set_diag
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax

from typing import Union, Tuple, Optional
from torch_geometric.typing import (OptPairTensor, Adj, Size, NoneType,
                                    OptTensor)

from torch import Tensor
import torch.nn.functional as F

from torch_geometric.nn import inits
import math

class Linear(torch.nn.Module):
    r"""Applies a linear tranformation to the incoming data
    .. math::
        \mathbf{x}^{\prime} = \mathbf{x} \mathbf{W}^{\top} + \mathbf{b}
    similar to :class:`torch.nn.Linear`.
    It supports lazy initialization and customizable weight and bias
    initialization.
    Args:
        in_channels (int): Size of each input sample. Will be initialized
            lazily in case it is given as :obj:`-1`.
        out_channels (int): Size of each output sample.
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        weight_initializer (str, optional): The initializer for the weight
            matrix (:obj:`"glorot"`, :obj:`"uniform"`, :obj:`"kaiming_uniform"`
            or :obj:`None`).
            If set to :obj:`None`, will match default weight initialization of
            :class:`torch.nn.Linear`. (default: :obj:`None`)
        bias_initializer (str, optional): The initializer for the bias vector
            (:obj:`"zeros"` or :obj:`None`).
            If set to :obj:`None`, will match default bias initialization of
            :class:`torch.nn.Linear`. (default: :obj:`None`)
    """
    def __init__(self, in_channels: int, out_channels: int, bias: bool = True,
                 weight_initializer: Optional[str] = None,
                 bias_initializer: Optional[str] = None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight_initializer = weight_initializer
        self.bias_initializer = bias_initializer

        if in_channels > 0:
            self.weight = Parameter(torch.Tensor(out_channels, in_channels))
        else:
            self.weight = nn.parameter.UninitializedParameter()
            self._hook = self.register_forward_pre_hook(
                self.initialize_parameters)

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self._load_hook = self._register_load_state_dict_pre_hook(
            self._lazy_load_hook)

        self.reset_parameters()

    def __deepcopy__(self, memo):
        out = Linear(self.in_channels, self.out_channels, self.bias
                     is not None, self.weight_initializer,
                     self.bias_initializer)
        if self.in_channels > 0:
            out.weight = copy.deepcopy(self.weight, memo)
        if self.bias is not None:
            out.bias = copy.deepcopy(self.bias, memo)
        return out

    def reset_parameters(self):
        if isinstance(self.weight, nn.parameter.UninitializedParameter):
            pass
        elif self.weight_initializer == 'glorot':
            inits.glorot(self.weight)
        elif self.weight_initializer == 'uniform':
            bound = 1.0 / math.sqrt(self.weight.size(-1))
            torch.nn.init.uniform_(self.weight.data, -bound, bound)
        elif self.weight_initializer == 'kaiming_uniform':
            inits.kaiming_uniform(self.weight, fan=self.in_channels,
                                  a=math.sqrt(5))
        elif self.weight_initializer is None:
            inits.kaiming_uniform(self.weight, fan=self.in_channels,
                                  a=math.sqrt(5))
        else:
            raise RuntimeError(f"Linear layer weight initializer "
                               f"'{self.weight_initializer}' is not supported")

        if isinstance(self.weight, nn.parameter.UninitializedParameter):
            pass
        elif self.bias is None:
            pass
        elif self.bias_initializer == 'zeros':
            inits.zeros(self.bias)
        elif self.bias_initializer is None:
            inits.uniform(self.in_channels, self.bias)
        else:
            raise RuntimeError(f"Linear layer bias initializer "
                               f"'{self.bias_initializer}' is not supported")

    def forward(self, x: Tensor) -> Tensor:
        """"""
        return F.linear(x, self.weight, self.bias)

    @torch.no_grad()
    def initialize_parameters(self, module, input):
        if isinstance(self.weight, nn.parameter.UninitializedParameter):
            self.in_channels = input[0].size(-1)
            self.weight.materialize((self.out_channels, self.in_channels))
            self.reset_parameters()
        self._hook.remove()
        delattr(self, '_hook')

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        if isinstance(self.weight, nn.parameter.UninitializedParameter):
            destination[prefix + 'weight'] = self.weight
        else:
            destination[prefix + 'weight'] = self.weight.detach()
        if self.bias is not None:
            destination[prefix + 'bias'] = self.bias.detach()

    def _lazy_load_hook(self, state_dict, prefix, local_metadata, strict,
                        missing_keys, unexpected_keys, error_msgs):

        weight = state_dict[prefix + 'weight']
        if isinstance(weight, nn.parameter.UninitializedParameter):
            self.in_channels = -1
            self.weight = nn.parameter.UninitializedParameter()
            if not hasattr(self, '_hook'):
                self._hook = self.register_forward_pre_hook(
                    self.initialize_parameters)

        elif isinstance(self.weight, nn.parameter.UninitializedParameter):
            self.in_channels = weight.size(-1)
            self.weight.materialize((self.out_channels, self.in_channels))
            if hasattr(self, '_hook'):
                self._hook.remove()
                delattr(self, '_hook')

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, bias={self.bias is not None})')

    
class GATConv(MessagePassing):
    r"""The graph attentional operator from the `"Graph Attention Networks"
    <https://arxiv.org/abs/1710.10903>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i = \alpha_{i,i}\mathbf{\Theta}\mathbf{x}_{i} +
        \sum_{j \in \mathcal{N}(i)} \alpha_{i,j}\mathbf{\Theta}\mathbf{x}_{j},

    where the attention coefficients :math:`\alpha_{i,j}` are computed as

    .. math::
        \alpha_{i,j} =
        \frac{
        \exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}
        [\mathbf{\Theta}\mathbf{x}_i \, \Vert \, \mathbf{\Theta}\mathbf{x}_j]
        \right)\right)}
        {\sum_{k \in \mathcal{N}(i) \cup \{ i \}}
        \exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}
        [\mathbf{\Theta}\mathbf{x}_i \, \Vert \, \mathbf{\Theta}\mathbf{x}_k]
        \right)\right)}.

    If the graph has multi-dimensional edge features :math:`\mathbf{e}_{i,j}`,
    the attention coefficients :math:`\alpha_{i,j}` are computed as

    .. math::
        \alpha_{i,j} =
        \frac{
        \exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}
        [\mathbf{\Theta}\mathbf{x}_i \, \Vert \, \mathbf{\Theta}\mathbf{x}_j
        \, \Vert \, \mathbf{\Theta}_{e} \mathbf{e}_{i,j}]\right)\right)}
        {\sum_{k \in \mathcal{N}(i) \cup \{ i \}}
        \exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}
        [\mathbf{\Theta}\mathbf{x}_i \, \Vert \, \mathbf{\Theta}\mathbf{x}_k
        \, \Vert \, \mathbf{\Theta}_{e} \mathbf{e}_{i,k}]\right)\right)}.

    Args:
        in_channels (int or tuple): Size of each input sample, or :obj:`-1` to
            derive the size from the first input(s) to the forward method.
            A tuple corresponds to the sizes of source and target
            dimensionalities.
        out_channels (int): Size of each output sample.
        heads (int, optional): Number of multi-head-attentions.
            (default: :obj:`1`)
        concat (bool, optional): If set to :obj:`False`, the multi-head
            attentions are averaged instead of concatenated.
            (default: :obj:`True`)
        negative_slope (float, optional): LeakyReLU angle of the negative
            slope. (default: :obj:`0.2`)
        dropout (float, optional): Dropout probability of the normalized
            attention coefficients which exposes each node to a stochastically
            sampled neighborhood during training. (default: :obj:`0`)
        add_self_loops (bool, optional): If set to :obj:`False`, will not add
            self-loops to the input graph. (default: :obj:`True`)
        edge_dim (int, optional): Edge feature dimensionality (in case
            there are any). (default: :obj:`None`)
        fill_value (float or Tensor or str, optional): The way to generate
            edge features of self-loops (in case :obj:`edge_dim != None`).
            If given as :obj:`float` or :class:`torch.Tensor`, edge features of
            self-loops will be directly given by :obj:`fill_value`.
            If given as :obj:`str`, edge features of self-loops are computed by
            aggregating all features of edges that point to the specific node,
            according to a reduce operation. (:obj:`"add"`, :obj:`"mean"`,
            :obj:`"min"`, :obj:`"max"`, :obj:`"mul"`). (default: :obj:`"mean"`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F_{in})` or
          :math:`((|\mathcal{V_s}|, F_{s}), (|\mathcal{V_t}|, F_{t}))`
          if bipartite,
          edge indices :math:`(2, |\mathcal{E}|)`,
          edge features :math:`(|\mathcal{E}|, D)` *(optional)*
        - **output:** node features :math:`(|\mathcal{V}|, H * F_{out})` or
          :math:`((|\mathcal{V}_t|, H * F_{out})` if bipartite.
          If :obj:`return_attention_weights=True`, then
          :math:`((|\mathcal{V}|, H * F_{out}),
          ((2, |\mathcal{E}|), (|\mathcal{E}|, H)))`
          or :math:`((|\mathcal{V_t}|, H * F_{out}), ((2, |\mathcal{E}|),
          (|\mathcal{E}|, H)))` if bipartite
    """
    _alpha: OptTensor

    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        negative_slope: float = 0.2,
        dropout: float = 0.2,
        add_self_loops: bool = True,
        edge_dim: Optional[int] = None,
        fill_value: Union[float, Tensor, str] = 'mean',
        bias: bool = True,
        **kwargs,
    ):
        kwargs.setdefault('aggr', 'mean')
        super().__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        self.edge_dim = edge_dim
        self.fill_value = fill_value

        self.norm = MessageNorm(True)

        # The learnable parameters to compute attention coefficients:
        self.att_src = Parameter(torch.Tensor(1, heads, out_channels))
        self.att_dst = Parameter(torch.Tensor(1, heads, out_channels))

        if edge_dim is not None:
            self.lin_edge = Linear(edge_dim, heads * out_channels, bias=False, weight_initializer='glorot')
            self.att_edge = Parameter(torch.Tensor(1, heads, out_channels))
        else:
            self.lin_edge = None
            self.register_parameter('att_edge', None)

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)

        self._alpha = None

        self.reset_parameters()

    def reset_parameters(self):
        if self.lin_edge is not None:
            self.lin_edge.reset_parameters()
        glorot(self.att_src)
        glorot(self.att_dst)
        glorot(self.att_edge)

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                edge_attr: OptTensor = None, size: Size = None,
                return_attention_weights=None):
        # type: (Union[Tensor, OptPairTensor], Tensor, OptTensor, Size, NoneType) -> Tensor  # noqa
        # type: (Union[Tensor, OptPairTensor], SparseTensor, OptTensor, Size, NoneType) -> Tensor  # noqa
        # type: (Union[Tensor, OptPairTensor], Tensor, OptTensor, Size, bool) -> Tuple[Tensor, Tuple[Tensor, Tensor]]  # noqa
        # type: (Union[Tensor, OptPairTensor], SparseTensor, OptTensor, Size, bool) -> Tuple[Tensor, SparseTensor]  # noqa
        r"""
        Args:
            return_attention_weights (bool, optional): If set to :obj:`True`,
                will additionally return the tuple
                :obj:`(edge_index, attention_weights)`, holding the computed
                attention weights for each edge. (default: :obj:`None`)
        """
        # NOTE: attention weights will be returned whenever
        # `return_attention_weights` is set to a value, regardless of its
        # actual value (might be `True` or `False`). This is a current somewhat
        # hacky workaround to allow for TorchScript support via the
        # `torch.jit._overload` decorator, as we can only change the output
        # arguments conditioned on type (`None` or `bool`), not based on its
        # actual value.

        H, C = self.heads, self.out_channels

        x_src = x_dst = x.view(-1, H, C)
        x = (x_src, x_dst)

        # Next, we compute node-level attention coefficients, both for source
        # and target nodes (if present):
        alpha_src = (x_src * self.att_src).sum(dim=-1)
        alpha_dst = None if x_dst is None else (x_dst * self.att_dst).sum(-1)
        alpha = (alpha_src, alpha_dst)
        

        # propagate_type: (x: OptPairTensor, alpha: OptPairTensor, edge_attr: OptTensor)  # noqa
        out = self.propagate(edge_index, x=x, alpha=alpha, edge_attr=edge_attr,
                             size=size)
        
        print(out)
        print(out.max())

        alpha = self._alpha
        assert alpha is not None
        self._alpha = None

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out += self.bias
            
        print(out)
        print(out.max())

        if isinstance(return_attention_weights, bool):
            if isinstance(edge_index, Tensor):
                return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out
        
    def update(self, inputs, x):
        return self.norm(x[0], inputs)

    def message(self, x_j: Tensor, alpha_j: Tensor, alpha_i: OptTensor,
                edge_attr: OptTensor, index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:
        # Given edge-level attention coefficients for source and target nodes,
        # we simply need to sum them up to "emulate" concatenation:
        alpha = alpha_j if alpha_i is None else alpha_j + alpha_i

        if edge_attr is not None:
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.view(-1, 1)
            assert self.lin_edge is not None
            edge_attr = self.lin_edge(edge_attr)
            edge_attr = edge_attr.view(-1, self.heads, self.out_channels)
            alpha_edge = (edge_attr * self.att_edge).sum(dim=-1)
            alpha = alpha + alpha_edge

        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha  # Save for later use.
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        
        return x_j * alpha.unsqueeze(-1)
    
    

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')
    
WIDTH = (np.sqrt(2) - 1) * 64 * 2
def design_lowpass_filter(numtaps = 5, cutoff = 64 - 1, width = WIDTH, fs = 128, radial=False):
    assert numtaps >= 1

    # Identity filter.
    if numtaps == 1:
        return None

    # Separable Kaiser low-pass filter.
    if not radial:
        f = scipy.signal.firwin(numtaps=numtaps, cutoff=cutoff, width=width, fs=fs)
        return torch.as_tensor(f, dtype=torch.float32)

    # Radially symmetric jinc-based filter.
    x = (np.arange(numtaps) - (numtaps - 1) / 2) / fs
    r = np.hypot(*np.meshgrid(x, x))
    f = scipy.special.j1(2 * cutoff * (np.pi * r)) / (np.pi * r)
    beta = scipy.signal.kaiser_beta(scipy.signal.kaiser_atten(numtaps, width / (fs / 2)))
    w = np.kaiser(numtaps, beta)
    f *= np.outer(w, w)
    f /= np.sum(f)
    return torch.as_tensor(f, dtype=torch.float32)
        

class Eq1dConv(nn.Module):
    def __init__(self, in_channels = 1, out_channels = 1, k = 3, s = 128):
        super(Eq1dConv, self).__init__()
        
        self.conv = nn.Conv2d(in_channels, out_channels, (1, k), 
                                        stride = (1, 1), padding = (0, (k + 1) // 2 - 1), bias = False)
        self.register_buffer('up_filter', design_lowpass_filter().view(1, 5))
        self.register_buffer('down_filter', design_lowpass_filter(4, fs = 256).view(1, 4))
        self.bias = torch.nn.Parameter(torch.zeros([out_channels]))
        
        self.conv_clamp = 64
        
        self.padding = [3, 3, 0, 0]
        
    def forward(self, x):
        # convolve and the perform
        x = self.conv(x)
        x = filtered_lrelu.filtered_lrelu(x=x, fu = self.up_filter, fd = self.down_filter, b = self.bias.to(x.dtype),
            up=2, down=2, padding=self.padding, gain=1., clamp=None)
        
        return x

class GCNConvNet_beta(nn.Module):
    def __init__(self, in_channels = 1, depth = 7, pred_pop = 1):
        super(GCNConvNet_beta, self).__init__()
        
        self.convs = nn.ModuleList()
        self.norm_convs = nn.ModuleList()
        self.gcns = nn.ModuleList()
        self.gcn_norms = nn.ModuleList()
        
        self.downs = nn.ModuleList()
        self.norms = nn.ModuleList()
    
        channels = 0
        for ix in range(depth):
            self.convs.append(Eq1dConv(in_channels, 1))
            self.gcns.append(GATConv(128, 128, edge_dim = 8))
            self.norms.append(nn.Sequential(nn.InstanceNorm2d(1), nn.Dropout2d(0.2)))
            self.gcn_norms.append(LayerNorm(128))
                    
            channels += 3
            in_channels = 3
            
            if ix > 0:
                self.downs.append(nn.Conv2d(3, 1, 1, 1))
            
        self.out = nn.Conv2d(channels, 1, 1, 1, bias = False)
        self.pred_pop = pred_pop
    
    def forward(self, x, edge_index, edge_attr, batch):
        #print(x.shape, edge_index.shape, edge_attr.shape, batch.shape)
        #print(edge_index.max())
        
        batch_size, _, ind, sites = x.shape
        xc = self.norms[0](self.convs[0](x))
        print(xc.max())
        
        xg = torch.flatten(xc.transpose(1, 2), 2, 3).flatten(0, 1)
        
        xg = self.gcn_norms[0](self.gcns[0](xg, edge_index, edge_attr))
        print(xg.max())
        
        xg = to_dense_batch(xg, batch)[0]
        xg = xg.reshape(batch_size, ind, 1, sites).transpose(1, 2)
        
        x = torch.cat([x, xc, xg], dim = 1)
        
        xs = [x]
        for ix in range(1, len(self.convs)):
            xc = self.norms[ix](self.convs[ix](xs[-1]))
            
            xg = torch.flatten(xc.transpose(1, 2), 2, 3).flatten(0, 1)
            xg = self.gcn_norms[ix](self.gcns[ix](xg, edge_index, edge_attr))
            
            xg = to_dense_batch(xg, batch)[0]
            xg = xg.reshape(batch_size, ind, 1, sites).transpose(1, 2)
                
            xs.append(torch.cat([self.downs[ix - 1](xs[-1]), xc, xg], dim = 1) + xs[-1])
                  
        x = torch.cat(xs, dim = 1)
        
        # we only want the second pop
        if self.pred_pop == 1:
            x = x[:,:,ind // 2:,:]
        # we only want the first pop
        elif self.pred_pop == 0:
            x = x[:,:,:ind // 2,:]
        
        x = self.out(x)
        
        return torch.squeeze(x)

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
    
    
    
    