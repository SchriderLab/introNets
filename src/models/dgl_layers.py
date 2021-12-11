# -*- coding: utf-8 -*-
import dgl
import torch.nn as nn
import torch

from torch_geometric.nn import MessageNorm

class Res1dEncoder(nn.Module):
    def __init__(self, in_channels = 1, n_res_layers = 4):
        super(Res1dEncoder, self).__init__()
        
        channels = [1, 24, 48, 96, 288]
        
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for ix in range(n_res_layers):
            self.convs.append(Res1dBlock(channels[ix], channels[ix + 1] // 3, 3))
            self.norms.append(nn.Sequential(nn.InstanceNorm2d(channels[ix + 1]), nn.Dropout2d(0.05)))
            
        self.out = nn.Conv2d(channels[-1], channels[-1] // 2, 1, 1)
            
    def forward(self, x):
        for conv, norm in zip(self.convs, self.norms):
            x = norm(conv(x)).relu_()
            
        x = self.out(x)
            
        return x
    
class Res1dBlockUp(nn.Module):
    def __init__(self, in_channels, out_channels, n_layers, 
                             k = 3, pooling = 'max', up = False):
        super(Res1dBlockUp, self).__init__()
        
        # pass each pop through their own convolutions
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for ix in range(n_layers):
            self.convs.append(nn.Conv2d(in_channels, out_channels, (1, 3), 
                                        stride = (1, 1), padding = (0, (k + 1) // 2 - 1), bias = False))

            self.norms.append(nn.Sequential(nn.InstanceNorm2d(out_channels), nn.Dropout2d(0.1)))
            
            in_channels = out_channels
        
        self.up = nn.Upsample(scale_factor=(1,2), mode='bicubic', align_corners=True)
        
        self.activation = nn.ReLU()
        
    def forward(self, x, edge_index, edge_attr, batch):
        x = self.up(x)
        
        batch_size, n_channels, n_ind, n_sites = x.shape
        xs = [self.norms[0](self.convs[0](x))]
        
        for ix in range(1, len(self.norms)):
            xs.append(self.convs[ix](xs[-1]))        
            xs[-1] = self.norms[ix](xs[-1] + xs[-2])
            
        x = torch.cat([x, self.activation(torch.cat(xs, dim = 1))], dim = 1)
        
        return x
    
    
class Res1dDecoder(nn.Module):
    def __init__(self, in_channels = 1, n_res_layers = 4):
        super(Res1dEncoder, self).__init__()
        
        channels = [288, 96, 48, 24, 1]
        
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for ix in range(n_res_layers):
            self.convs.append(Res1dBlockUp(channels[ix], channels[ix + 1] // 3, 3))
            self.norms.append(nn.Sequential(nn.InstanceNorm2d(channels[ix + 1]), nn.Dropout2d(0.05)))
            
    def forward(self, x):
        for conv, norm in zip(self.convs, self.norms):
            x = norm(conv(x))
            
        return x

class TreeLSTMCell(nn.Module):
    def __init__(self, h_size):
        super(TreeLSTMCell, self).__init__()
        self.U_iou = nn.Linear(2 * h_size, 3 * h_size, bias = False)
        self.b_iou = nn.Parameter(torch.zeros(1, 3 * h_size))
        self.U_f = nn.Linear(2 * h_size, 2 * h_size)
        self.norm_iou = MessageNorm(True)
        self.norm_c = MessageNorm(True)

    def message_func(self, edges):
        return {'h': edges.src['h'], 'c': edges.src['c']}

    def reduce_func(self, nodes):
        # concatenate h_jl for equation (1), (2), (3), (4)
        #print(nodes.mailbox['h'].shape)
        h_cat = nodes.mailbox['h'].view(nodes.mailbox['h'].size(0), -1)
        #print(h_cat.shape)
        
        # equation (2)
        f = torch.sigmoid(self.U_f(h_cat)).view(*nodes.mailbox['h'].size())
        # second term of equation (5)
        c = torch.sum(f * nodes.mailbox['c'], 1)
        return {'iou': self.U_iou(self.norm_iou(nodes.data['iou'], h_cat)), 'c': self.norm_c(nodes.mailbox['c'][:,0,:], c)}

    def apply_node_func(self, nodes):
        # equation (1), (3), (4)
        iou = nodes.data['iou'] + self.b_iou
        i, o, u = torch.chunk(iou, 3, 1)
        i, o, u = torch.sigmoid(i), torch.sigmoid(o), torch.tanh(u)
        # equation (5)
        c = i * u + nodes.data['c']
        # equation (6)
        h = o * torch.tanh(c)
        return {'h' : h, 'c' : c}

from layers_1d import Res1dBlock
import copy

class DownConvTransform(nn.Module):
    def __init__(self, in_channels):
        super(DownConvTransform, self).__init__()
        self.conv = nn.Conv2d()

class TreeLSTM(nn.Module):
    def __init__(self, in_channels = 1, conv_channels = 48, n_cycles = 2, h_size = 128, dropout = 0.1):
        super(TreeLSTM, self).__init__()
        
        self.encoder = Res1dEncoder()
        self.decoder = Res1dDecoder()
        
        # at each iteration of topological message passing, convolve the node features across
            
        self.dropout = nn.Dropout(dropout)
        self.cell = TreeLSTMCell(256)

    def forward(self, g, h, c):
        """Compute tree-lstm prediction given a batch.

        Parameters
        ----------
        batch : dgl.data.SSTBatch
            The data batch.
        h : Tensor
            Initial hidden state.
        c : Tensor
            Initial cell state.

        Returns
        -------
        logits : Tensor
            The prediction of each node.
        """
        
        ind, s = g.ndata['x'].shape
        
        # 1d image
        g.ndata['x'] = (g.ndata['x']).view(ind, 1, 1, s)
        
        # transform all the nodes via 1d convolution
        g.ndata['iou'] = self.encoder(g.ndata['x'])

        g.ndata['h'] = h
        g.ndata['c'] = c
        reverse = False
        
        dgl.prop_nodes_topo(g,
                            message_func=self.cell.message_func,
                            reduce_func=self.cell.reduce_func,
                            apply_node_func=self.cell.apply_node_func, reverse = reverse)
        g.ndata['iou'] = g.ndata['iou'].relu_()

        
                          
        for ix in range(1, len(self.convs)):
            reverse = (not reverse)
            
            g.ndata['iou'] = self.down_convs[ix](self.norms[ix](self.convs[ix](g.ndata['iou'].view(ind, 3, 1, s)))).view(ind, 3 * s)
            dgl.prop_nodes_topo(g,
                            message_func=self.cell.message_func,
                            reduce_func=self.cell.reduce_func,
                            apply_node_func=self.cell.apply_node_func, reverse = reverse)
            
            g.ndata['iou'] = g.ndata['iou'].relu_()
            xs.append(g.ndata['iou'].view(ind, 3, 1, s))
            
        # propagate
        dgl.prop_nodes_topo(g,
                            message_func=self.cell.message_func,
                            reduce_func=self.cell.reduce_func,
                            apply_node_func=self.cell.apply_node_func)
        # compute logits
        h = self.dropout(g.ndata.pop('h'))
        # take the hidden state, all the ious that we're convolved and concatenate
        
        # 3 * (n_convs + 1) channels
        xs = torch.cat(xs + [h.view(ind, 1, 1, s)], dim = 1)

        xs = torch.squeeze(self.out(xs))
        return xs

