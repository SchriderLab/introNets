# -*- coding: utf-8 -*-
import dgl
import torch.nn as nn
import torch

class TreeLSTMCell(nn.Module):
    def __init__(self, h_size):
        super(TreeLSTMCell, self).__init__()
        self.U_iou = nn.Linear(2 * h_size, 3 * h_size, bias=False)
        self.b_iou = nn.Parameter(torch.zeros(1, 3 * h_size))
        self.U_f = nn.Linear(2 * h_size, 2 * h_size)

    def message_func(self, edges):
        return {'h': edges.src['h'], 'c': edges.src['c']}

    def reduce_func(self, nodes):
        # concatenate h_jl for equation (1), (2), (3), (4)
        h_cat = nodes.mailbox['h'].view(nodes.mailbox['h'].size(0), -1)
        # equation (2)
        f = torch.sigmoid(self.U_f(h_cat)).view(*nodes.mailbox['h'].size())
        # second term of equation (5)
        c = torch.sum(f * nodes.mailbox['c'], 1)
        return {'iou': self.U_iou(h_cat), 'c': c}

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
        
        n_convs = n_cycles * 2
        
        self.init_conv = Res1dBlock((1,), 1, 3)
        self.init_norm = nn.InstanceNorm2d(3)
        
        self.convs = nn.ModuleList()
        self.down_convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        channels = in_channels
        
        # at each iteration of topological message passing, convolve the node features across
        # the chromosome.
        for ix in range(n_convs):
            self.convs.append(Res1dBlock((channels,), conv_channels // 3, n_layers = 3))
            self.down_convs.append(nn.Conv2d(conv_channels, 3, 1, bias = False))
            self.norms.append(nn.InstanceNorm2d(conv_channels))
            
            channels = conv_channels
            
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Conv2d(n_convs * 3 + 1, 1, 1)
        self.cell = TreeLSTMCell(h_size)

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
        g.ndata['x'] = (g.ndata['x']).view(g.ndata['x'].shape[0], 1, 1, g.ndata['x'].shape[0])
        
        # transform all the nodes via 1d convolution
        g.ndata['iou'] = self.init_norm(self.init_conv(g.ndata['x'])).view(ind, 3 * s)

        g.ndata['h'] = h
        g.ndata['c'] = c
        reverse = False
        
        dgl.prop_nodes_topo(g,
                            message_func=self.cell.message_func,
                            reduce_func=self.cell.reduce_func,
                            apply_node_func=self.cell.apply_node_func, reverse = reverse)
        xs = [g.ndata['iou'].view(ind, 3, 1, s)]
                          
        for ix in range(1, len(self.convs)):
            reverse = (not reverse)
            
            g.ndata['iou'] = self.down_convs[ix](self.norms[ix](self.convs[ix](g.ndata['iou'].view(ind, 3, 1, s))).relu_()).relu_().view(ind, 3 * s)
            dgl.prop_nodes_topo(g,
                            message_func=self.cell.message_func,
                            reduce_func=self.cell.reduce_func,
                            apply_node_func=self.cell.apply_node_func, reverse = reverse)
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

        xs = self.out(xs)
        return xs

