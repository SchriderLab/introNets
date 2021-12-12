# -*- coding: utf-8 -*-
import dgl
import torch.nn as nn
import torch

from torch_geometric.nn import MessageNorm

class Res1dEncoder(nn.Module):
    def __init__(self, in_channels = 1, n_res_layers = 4):
        super(Res1dEncoder, self).__init__()
        
        channels = [1, 12, 24, 48, 96]
        
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for ix in range(n_res_layers):
            self.convs.append(Res1dBlock((channels[ix],), channels[ix + 1] // 3, 3))
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
        
    def forward(self, x):
        x = self.up(x)
        
        batch_size, n_channels, n_ind, n_sites = x.shape
        xs = [self.norms[0](self.convs[0](x))]
        
        for ix in range(1, len(self.norms)):
            xs.append(self.convs[ix](xs[-1]))        
            xs[-1] = self.norms[ix](xs[-1] + xs[-2])
            
        x = self.activation(torch.cat(xs, dim = 1))
        
        return x
    
    
class Res1dDecoder(nn.Module):
    def __init__(self, in_channels = 1, n_res_layers = 4):
        super(Res1dDecoder, self).__init__()
        
        channels = [48, 48, 48, 24, 3]
        
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for ix in range(n_res_layers):
            self.convs.append(Res1dBlockUp(channels[ix], channels[ix + 1] // 3, 3))
            self.norms.append(nn.Sequential(nn.InstanceNorm2d(channels[ix + 1]), nn.Dropout2d(0.05)))
            
    def forward(self, x):
        for conv, norm in zip(self.convs, self.norms):
            x = norm(conv(x))
            
        return x
    
class Attention_block(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super(Attention_block,self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
            )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self,g,x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)

        return x*psi
    
class TreeResUNet(nn.Module):
    def __init__(self, n_layers = 4):
        super(TreeResUNet, self).__init__()
        channels = [1, 9, 27, 48, 96]
        self.h_sizes = [192, 192, 192, 192]
        in_sizes = [288, 432, 384, 384]
        
        self.h_mlp = nn.Sequential(nn.Linear(4, 16), nn.LayerNorm(16),
                                   nn.ReLU(), nn.Linear(16, 64), nn.LayerNorm(64), nn.ReLU(), 
                                   nn.Linear(64, 192))
        
        self.down_convs = nn.ModuleList()
        self.down_norms = nn.ModuleList()
        
        self.c0 = Res1dBlock((1,), 3, 3)
        
        self.down_transforms = nn.ModuleList()
        self.down_lstms = nn.ModuleList()
        self.down_ls_norms = nn.ModuleList()
        
        for ix in range(len(channels) - 1):
            if ix != 0:
                self.down_convs.append(Res1dBlock((channels[ix],), channels[ix + 1] // 3, 3))
            self.down_transforms.append(nn.Sequential(nn.Linear(in_sizes[ix], self.h_sizes[ix] * 3), nn.LayerNorm(self.h_sizes[ix] * 3)))
            self.down_norms.append(nn.InstanceNorm2d(channels[ix + 1]))
            
            self.down_ls_norms.append(nn.LayerNorm(768))
            
            self.down_lstms.append(TreeLSTMCell(self.h_sizes[ix]))
            
        self.up_convs = nn.ModuleList()
        self.up_norms = nn.ModuleList()
            
        # left side of the u
        channels = [96, 48, 27, 9, 9]
        for ix in range(len(channels) - 1):
            self.up_convs.append(Res1dBlockUp(channels[ix], channels[ix + 1], 1))
            self.up_norms.append(nn.InstanceNorm2d(channels[ix + 1] * 2 + 24))
            
            channels[ix + 1] = channels[ix + 1] * 2 + 24

        self.up1_0_lstm = nn.Sequential(Res1dBlock((12,), 12, 2, pooling = None), nn.Dropout2d(0.1))
        self.up2_1_lstm = nn.Sequential(Res1dBlock((12,), 12, 2), nn.Dropout2d(0.1))

        self.up3_2_lstm = nn.Sequential(Res1dBlock((12,), 12, 2), nn.Dropout2d(0.1), Res1dBlock((24,), 24, 1), nn.Dropout2d(0.1))        
        self.up4_3_lstm = nn.Sequential(Res1dBlock((12,), 12, 2), nn.Dropout2d(0.1), 
                                        Res1dBlock((24,), 24, 1), nn.Dropout2d(0.1), 
                                        Res1dBlock((24,), 24, 1), nn.Dropout2d(0.1))
        
        self.out = nn.Conv2d(43, 1, 1)
        
    def forward(self, g, h):
        ind, s = g.ndata['x'].shape
        
        # 1d image
        g.ndata['x'] = (g.ndata['x']).view(ind, 1, 1, s)
        xs = [g.ndata['x']]
        vs = []
        
        # go down
        x, x_ = self.c0(xs[-1], return_unpooled = True)
        x = self.down_norms[0](x)
        xs.append(x)
        
        xs[0] = torch.cat([x_, xs[0]], dim = 1)
        
        g.ndata['h'] = self.h_mlp(h)
        g.ndata['c'] = torch.zeros((ind, self.h_sizes[0])).to(torch.device('cuda'))
        
        g.ndata['iou'] = self.down_transforms[0](x.view(ind, -1))
        
        dgl.prop_nodes_topo(g,
                            message_func=self.down_lstms[0].message_func,
                            reduce_func=self.down_lstms[0].reduce_func,
                            apply_node_func=self.down_lstms[0].apply_node_func, reverse = False)
        
        dgl.prop_nodes_topo(g,
                            message_func=self.down_lstms[0].message_func,
                            reduce_func=self.down_lstms[0].reduce_func,
                            apply_node_func=self.down_lstms[0].apply_node_func, reverse = True)
        vs.append(self.down_ls_norms[0](torch.cat([g.ndata.pop('h'), g.ndata.pop('iou')], dim = 1)))
        
        for ix in range(1, len(self.down_convs)):
            # go down
            x = self.down_norms[ix](self.down_convs[ix](xs[-1]))
            
            g.ndata['h'] = self.h_mlp(h)
            g.ndata['c'] = torch.zeros((ind, self.h_sizes[ix])).to(torch.device('cuda'))
            
            g.ndata['iou'] = self.down_transforms[ix](x.view(ind, -1))
            
            dgl.prop_nodes_topo(g,
                                message_func=self.down_lstms[ix].message_func,
                                reduce_func=self.down_lstms[ix].reduce_func,
                                apply_node_func=self.down_lstms[ix].apply_node_func, reverse = False)
            
            dgl.prop_nodes_topo(g,
                                message_func=self.down_lstms[ix].message_func,
                                reduce_func=self.down_lstms[ix].reduce_func,
                                apply_node_func=self.down_lstms[ix].apply_node_func, reverse = True)
            
            xs.append(x)
            vs.append(self.down_ls_norms[ix](torch.cat([g.ndata.pop('h'), g.ndata.pop('iou')], dim = 1)))
            
        # go back up for the lstm
        vs[-1] = self.up4_3_lstm(vs[-1].view(ind, 12, 1, 64))
        xs[-2] = self.up_norms[0](torch.cat([xs[-2], vs[-1], self.up_convs[0](xs[-1])], dim = 1))

        vs[-2] = self.up3_2_lstm(vs[-2].view(ind, 12, 1, 64))
        xs[-3] = self.up_norms[1](torch.cat([xs[-3], vs[-2], self.up_convs[1](xs[-2])], dim = 1))
        
        vs[-3] = self.up2_1_lstm(vs[-3].view(ind, 12, 1, 64))
        xs[-4] = self.up_norms[2](torch.cat([xs[-4], vs[-3], self.up_convs[2](xs[-3])], dim = 1))
        
        vs[-4] = self.up1_0_lstm(vs[-4].view(ind, 12, 1, 64))
        xs[-5] = self.up_norms[3](torch.cat([xs[-5], vs[-4], self.up_convs[3](xs[-4])], dim = 1))
        
        del vs
        
        return torch.squeeze(self.out(xs[-5]))
            
            
        
        

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
        i, o, u = torch.relu(i), torch.sigmoid(o), torch.tanh(u)
        
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
        self.out = nn.Conv2d(3, 1, 1, 1)
        
        self.cell = TreeLSTMCell(384)

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
        g.ndata['iou'] = g.ndata['iou'].view(ind, -1)
        
        g.ndata['h'] = h
        g.ndata['c'] = c
        
        dgl.prop_nodes_topo(g,
                            message_func=self.cell.message_func,
                            reduce_func=self.cell.reduce_func,
                            apply_node_func=self.cell.apply_node_func, reverse = False)
        
        dgl.prop_nodes_topo(g,
                            message_func=self.cell.message_func,
                            reduce_func=self.cell.reduce_func,
                            apply_node_func=self.cell.apply_node_func, reverse = True)

        # compute logits
        h = self.dropout(g.ndata.pop('h')).view(ind, 48, 1, 8)
        h = self.decoder(h)
        # take the hidden state, all the ious that we're convolved and concatenate
        
        h = torch.squeeze(self.out(h))
        return h

