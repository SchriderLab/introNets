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

#from sparsenn.models.gcn.layers import DynamicGraphResBlock
#from torch_geometric.nn import global_mean_pool

InceptionOutputs = namedtuple('InceptionOutputs', ['logits', 'aux_logits'])
InceptionOutputs.__annotations__ = {'logits': Tensor, 'aux_logits': Optional[Tensor]}

#from torch_scatter import scatter_max, scatter, scatter_mean, scatter_std

# Script annotations failed with _GoogleNetOutputs = namedtuple ...
# _InceptionOutputs set here for backwards compat
_InceptionOutputs = InceptionOutputs
#from torch_geometric.utils import to_dense_batch

class LSTMClassifier(nn.Module):

    def __init__(self, vocab_size = 2, input_dim = 34, hidden_dim = 512, output_size = 3):

        super(LSTMClassifier, self).__init__()

        self.hidden_dim = hidden_dim

        self.lstm = nn.GRU(input_dim, hidden_dim, num_layers = 1, batch_first = True)

        self.hidden2out = nn.Linear(hidden_dim, output_size)
        self.softmax = nn.LogSoftmax(dim = -1)

        self.dropout_layer = nn.Dropout(p=0.2)


    def init_hidden(self, batch_size):
        return(autograd.Variable(torch.randn(1, batch_size, self.hidden_dim)),
                        autograd.Variable(torch.randn(1, batch_size, self.hidden_dim)))


    def forward(self, x):
        self.hidden = self.init_hidden(x.size(0))
        self.hidden = (self.hidden[0].to(x.device), self.hidden[1].to(x.device))

        outputs, (ht, ct) = self.lstm(x, self.hidden)

        # ht is the last hidden state of the sequences
        # ht = (1 x batch_size x hidden_dim)
        # ht[-1] = (batch_size x hidden_dim)
        output = self.dropout_layer(ht[-1])
        output = self.hidden2out(output)
        output = self.softmax(output)

        return output
        
class LexStyleNet(nn.Module):
    def __init__(self, h = 34, w = 508, n_layers = 3):
        super(LexStyleNet, self).__init__()

        self.convs = nn.ModuleList()
        
        self.down = nn.MaxPool1d(2)
        
        in_channels = h
        out_channels = [48, 96, 128]
        for ix in range(n_layers):
            self.convs.append(nn.Sequential(nn.Conv1d(in_channels, out_channels[ix], 5, padding = 2), nn.InstanceNorm1d(out_channels[ix]), nn.ReLU(), nn.Dropout(0.1)))
            
            in_channels = copy.copy(out_channels[ix])
            
            w = w // 2
        
        
        features = 4
        
        self.out_size = out_channels[-1] * features
        self.out = nn.Sequential(nn.Linear(self.out_size, self.out_size // 2), nn.LayerNorm((self.out_size // 2,)), nn.ReLU(), 
                                 nn.Linear(self.out_size // 2, self.out_size // 4), nn.LayerNorm((self.out_size // 4,)), nn.ReLU(),
                                 nn.Linear(self.out_size // 4, 3), nn.LogSoftmax(dim = -1)) 
        
    def forward(self, x):
        for ix in range(len(self.convs)):
            x = self.convs[ix](x)
            x = self.down(x)
        
        xm = x.mean(dim = -1)
        xs = x.std(dim = -1)
        xmax = x.max(dim = -1)
        xmin = x.min(dim = -1)
        
        x = torch.cat([xm, xs, xmax, xmin], dim = -1)
        
        return self.out(x)
        
            

class GCNUNet_i1(nn.Module):
    def __init__(self, in_channels = 256, n_classes = 1, n_features = 256, n_layers = 8, layer_type = 'gat'):
        super(GCNUNet_i1, self).__init__()
        
        self.res = DynamicGraphResBlock(in_channels, n_features, n_layers, layer_type = layer_type)
        n = n_features * (n_layers - 1)
        
        self.conv = nn.Conv1d(n, 1024, 1)
        self.transform = nn.Sequential(nn.Linear(n + 1024, 1024), nn.BatchNorm1d(1024), nn.ReLU(), 
                                       nn.Linear(1024, 1024), nn.BatchNorm1d(1024), nn.ReLU(), 
                                       nn.Linear(1024, n_classes))
        
        self.activation = nn.ReLU()
        
    def forward(self, x, edge_indices, batch):
        x = self.res(x, edge_indices)
        
        x_global = self.conv(torch.unsqueeze(x, 2))
        x_global = torch.squeeze(self.activation(x_global))
        
        x_global_max = scatter_max(x_global, batch, dim = 0)[0]
        
        x = torch.cat([x, 
                       x_global_max[batch]], dim = 1)
        
        x = self.transform(x)
        
        return x

class GCNUNet_i2(nn.Module):
    def __init__(self, in_channels = 256, n_classes = 1, 
                 n_features = 256, n_layers = 8, n_global = 1024, 
                 layer_type = 'gat', n_heads = 2, return_attention_weights = False):
        super(GCNUNet_i2, self).__init__()
        
        self.res = DynamicGraphResBlock(in_channels, n_features, n_layers, 
                                        layer_type = layer_type, heads = n_heads, return_attention_weights = return_attention_weights)
        n = n_features * (n_layers - 1)
        
        self.conv = nn.Conv1d(n, n_global, 1)
        self.transform = nn.Sequential(nn.Linear(n + 3 * n_global, 2048), nn.LayerNorm(2048), nn.ReLU(), 
                                       nn.Linear(2048, 2048), nn.LayerNorm(2048), nn.ReLU(), 
                                       nn.Linear(2048, n_classes))
        
        self.activation = nn.ReLU()
        
    def forward(self, x, edge_indices, batch):
        x = self.res(x, edge_indices)
        
        x_global = self.conv(torch.unsqueeze(x, 2))
        x_global = torch.squeeze(self.activation(x_global))
        
        x_global_max = scatter_max(x_global, batch, dim = 0)[0]
        x_global_mean = scatter_mean(x_global, batch, dim = 0)
        x_global_std = scatter_std(x_global, batch, dim = 0)
        
        x = torch.cat([x, 
                       x_global_max[batch], 
                       x_global_mean[batch], 
                       x_global_std[batch]], dim = 1)
        
        x = self.transform(x)
        
        return x
    
class GCNUNet_i3(nn.Module):
    def __init__(self, in_channels = 256, n_classes = 1, 
                 n_features = 256, n_layers = 8, n_global = 1024, 
                 layer_type = 'gat', n_heads = 2, return_attention_weights = False):
        super(GCNUNet_i3, self).__init__()
        
        self.res = DynamicGraphResBlock(in_channels, n_features, n_layers, 
                                        layer_type = layer_type, heads = n_heads, return_attention_weights = return_attention_weights)
        n = n_features * (n_layers - 1)
        
        self.transform = nn.Sequential(nn.Linear(2 * n, 4096), nn.LayerNorm(4096), nn.ELU(), nn.Dropout(0.25),
                                       nn.Linear(4096, 4096), nn.LayerNorm(4096), nn.ELU(), nn.Dropout(0.25),
                                       nn.Linear(4096, n_classes))
        
        self.activation = nn.ReLU()
        
    def forward(self, x, edge_indices, batch):
        x = self.res(x, edge_indices)
        
        x_global = torch.squeeze(x)
        x_global = scatter_max(x_global, batch, dim = 0)[0]
        
        x = torch.cat([x, x_global[batch]], dim = 1)
        
        x = self.transform(x)
        
        return x
    
class GCNClassifier(nn.Module):
    def __init__(self, in_channels = 306, n_classes = 4, n_features = 306, n_layers = 8):
        super(GCNClassifier, self).__init__()
        
        self.res = DynamicGraphResBlock(in_channels, n_features, n_layers)
        n = n_features * (n_layers - 1)
        
        self.conv = nn.Conv1d(n, 1024, 1)
        self.transform = nn.Sequential(nn.Linear(n + 1024, 1024), nn.BatchNorm1d(1024), nn.ReLU(), 
                                       nn.Linear(1024, 1024), nn.BatchNorm1d(1024), nn.ReLU(), 
                                       nn.Linear(1024, n_classes))
        
        self.activation = nn.ReLU()
        
    def forward(self, x, edge_indices, batch):
        x = self.res(x, edge_indices)
        
        x_global = self.conv(torch.unsqueeze(x, 2))
        x_global = torch.squeeze(self.activation(x_global))
        
        x_global = scatter_max(x_global, batch, dim = 0)[0]
        
        x = torch.cat([x, x_global[batch]], dim = 1)
        x = global_mean_pool(x, batch)
        
        x = self.transform(x)
        
        return x

class Inception3(nn.Module):
    def __init__(
        self,
        num_classes: int = 1000,
        aux_logits: bool = True,
        transform_input: bool = False,
        inception_blocks: Optional[List[Callable[..., nn.Module]]] = None,
        init_weights: Optional[bool] = None,
        in_channels: int = 3
    ) -> None:
        super(Inception3, self).__init__()
        if inception_blocks is None:
            inception_blocks = [
                BasicConv2d, InceptionA, InceptionB, InceptionC,
                InceptionD, InceptionE, InceptionAux
            ]
        if init_weights is None:
            warnings.warn('The default weight initialization of inception_v3 will be changed in future releases of '
                          'torchvision. If you wish to keep the old behavior (which leads to long initialization times'
                          ' due to scipy/scipy#11299), please set init_weights=True.', FutureWarning)
            init_weights = True
        assert len(inception_blocks) == 7
        conv_block = inception_blocks[0]
        inception_a = inception_blocks[1]
        inception_b = inception_blocks[2]
        inception_c = inception_blocks[3]
        inception_d = inception_blocks[4]
        inception_e = inception_blocks[5]
        inception_aux = inception_blocks[6]

        self.aux_logits = aux_logits
        self.transform_input = transform_input
        self.Conv2d_1a_3x3 = conv_block(in_channels, 32, kernel_size=3, stride=2)
        self.Conv2d_2a_3x3 = conv_block(32, 32, kernel_size=3)
        self.Conv2d_2b_3x3 = conv_block(32, 64, kernel_size=3, padding=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.Conv2d_3b_1x1 = conv_block(64, 80, kernel_size=1)
        self.Conv2d_4a_3x3 = conv_block(80, 192, kernel_size=3)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.Mixed_5b = inception_a(192, pool_features=32)
        self.Mixed_5c = inception_a(256, pool_features=64)
        self.Mixed_5d = inception_a(288, pool_features=64)
        self.Mixed_6a = inception_b(288)
        self.Mixed_6b = inception_c(768, channels_7x7=128)
        self.Mixed_6c = inception_c(768, channels_7x7=160)
        self.Mixed_6d = inception_c(768, channels_7x7=160)
        self.Mixed_6e = inception_c(768, channels_7x7=192)
        self.AuxLogits: Optional[nn.Module] = None
        if aux_logits:
            self.AuxLogits = inception_aux(768, num_classes)
        self.Mixed_7a = inception_d(768)
        self.Mixed_7b = inception_e(1280)
        self.Mixed_7c = inception_e(2048)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout()
        self.fc = nn.Linear(2048, num_classes)
        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                    stddev = float(m.stddev) if hasattr(m, 'stddev') else 0.1  # type: ignore
                    torch.nn.init.trunc_normal_(m.weight, mean=0.0, std=stddev, a=-2, b=2)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def _transform_input(self, x: Tensor) -> Tensor:
        if self.transform_input:
            x_ch0 = torch.unsqueeze(x[:, 0], 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x_ch1 = torch.unsqueeze(x[:, 1], 1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x_ch2 = torch.unsqueeze(x[:, 2], 1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
            x = torch.cat((x_ch0, x_ch1, x_ch2), 1)
        return x

    def _forward(self, x: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        # N x 3 x 299 x 299
        x = self.Conv2d_1a_3x3(x)
        # N x 32 x 149 x 149
        x = self.Conv2d_2a_3x3(x)
        # N x 32 x 147 x 147
        x = self.Conv2d_2b_3x3(x)
        # N x 64 x 147 x 147
        x = self.maxpool1(x)
        # N x 64 x 73 x 73
        x = self.Conv2d_3b_1x1(x)
        # N x 80 x 73 x 73
        x = self.Conv2d_4a_3x3(x)
        # N x 192 x 71 x 71
        x = self.maxpool2(x)
        # N x 192 x 35 x 35
        x = self.Mixed_5b(x)
        # N x 256 x 35 x 35
        x = self.Mixed_5c(x)
        # N x 288 x 35 x 35
        x = self.Mixed_5d(x)
        # N x 288 x 35 x 35
        x = self.Mixed_6a(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6b(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6c(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6d(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6e(x)
        # N x 768 x 17 x 17
        aux: Optional[Tensor] = None
        if self.AuxLogits is not None:
            if self.training:
                aux = self.AuxLogits(x)
        # N x 768 x 17 x 17
        x = self.Mixed_7a(x)
        # N x 1280 x 8 x 8
        x = self.Mixed_7b(x)
        # N x 2048 x 8 x 8
        x = self.Mixed_7c(x)
        # N x 2048 x 8 x 8
        # Adaptive average pooling
        x = self.avgpool(x)
        # N x 2048 x 1 x 1
        x = self.dropout(x)
        # N x 2048 x 1 x 1
        x = torch.flatten(x, 1)
        # N x 2048
        x = self.fc(x)
        # N x 1000 (num_classes)
        return x, aux

    @torch.jit.unused
    def eager_outputs(self, x: Tensor, aux: Optional[Tensor]):
        if self.training and self.aux_logits:
            return InceptionOutputs(x, aux)
        else:
            return x  # type: ignore[return-value]

    def forward(self, x: Tensor):
        x = self._transform_input(x)
        x, aux = self._forward(x)
        aux_defined = self.training and self.aux_logits
        if torch.jit.is_scripting():
            if not aux_defined:
                warnings.warn("Scripted Inception3 always returns Inception3 Tuple")
            return InceptionOutputs(x, aux)
        else:
            return self.eager_outputs(x, aux)


class InceptionA(nn.Module):

    def __init__(
        self,
        in_channels: int,
        pool_features: int,
        conv_block: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(InceptionA, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch1x1 = conv_block(in_channels, 64, kernel_size=1)

        self.branch5x5_1 = conv_block(in_channels, 48, kernel_size=1)
        self.branch5x5_2 = conv_block(48, 64, kernel_size=5, padding=2)

        self.branch3x3dbl_1 = conv_block(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = conv_block(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = conv_block(96, 96, kernel_size=3, padding=1)

        self.branch_pool = conv_block(in_channels, pool_features, kernel_size=1)

    def _forward(self, x: Tensor) -> List[Tensor]:
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return outputs

    def forward(self, x: Tensor) -> Tensor:
        outputs = self._forward(x)
        return torch.cat(outputs, 1)


class InceptionB(nn.Module):

    def __init__(
        self,
        in_channels: int,
        conv_block: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(InceptionB, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch3x3 = conv_block(in_channels, 384, kernel_size=3, stride=2)

        self.branch3x3dbl_1 = conv_block(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = conv_block(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = conv_block(96, 96, kernel_size=3, stride=2)

    def _forward(self, x: Tensor) -> List[Tensor]:
        branch3x3 = self.branch3x3(x)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)

        outputs = [branch3x3, branch3x3dbl, branch_pool]
        return outputs

    def forward(self, x: Tensor) -> Tensor:
        outputs = self._forward(x)
        return torch.cat(outputs, 1)


class InceptionC(nn.Module):

    def __init__(
        self,
        in_channels: int,
        channels_7x7: int,
        conv_block: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(InceptionC, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch1x1 = conv_block(in_channels, 192, kernel_size=1)

        c7 = channels_7x7
        self.branch7x7_1 = conv_block(in_channels, c7, kernel_size=1)
        self.branch7x7_2 = conv_block(c7, c7, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7_3 = conv_block(c7, 192, kernel_size=(7, 1), padding=(3, 0))

        self.branch7x7dbl_1 = conv_block(in_channels, c7, kernel_size=1)
        self.branch7x7dbl_2 = conv_block(c7, c7, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_3 = conv_block(c7, c7, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7dbl_4 = conv_block(c7, c7, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_5 = conv_block(c7, 192, kernel_size=(1, 7), padding=(0, 3))

        self.branch_pool = conv_block(in_channels, 192, kernel_size=1)

    def _forward(self, x: Tensor) -> List[Tensor]:
        branch1x1 = self.branch1x1(x)

        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)

        branch7x7dbl = self.branch7x7dbl_1(x)
        branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch7x7, branch7x7dbl, branch_pool]
        return outputs

    def forward(self, x: Tensor) -> Tensor:
        outputs = self._forward(x)
        return torch.cat(outputs, 1)


class InceptionD(nn.Module):

    def __init__(
        self,
        in_channels: int,
        conv_block: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(InceptionD, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch3x3_1 = conv_block(in_channels, 192, kernel_size=1)
        self.branch3x3_2 = conv_block(192, 320, kernel_size=3, stride=2)

        self.branch7x7x3_1 = conv_block(in_channels, 192, kernel_size=1)
        self.branch7x7x3_2 = conv_block(192, 192, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7x3_3 = conv_block(192, 192, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7x3_4 = conv_block(192, 192, kernel_size=3, stride=2)

    def _forward(self, x: Tensor) -> List[Tensor]:
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)

        branch7x7x3 = self.branch7x7x3_1(x)
        branch7x7x3 = self.branch7x7x3_2(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_3(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_4(branch7x7x3)

        branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)
        outputs = [branch3x3, branch7x7x3, branch_pool]
        return outputs

    def forward(self, x: Tensor) -> Tensor:
        outputs = self._forward(x)
        return torch.cat(outputs, 1)


class InceptionE(nn.Module):

    def __init__(
        self,
        in_channels: int,
        conv_block: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(InceptionE, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch1x1 = conv_block(in_channels, 320, kernel_size=1)

        self.branch3x3_1 = conv_block(in_channels, 384, kernel_size=1)
        self.branch3x3_2a = conv_block(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3_2b = conv_block(384, 384, kernel_size=(3, 1), padding=(1, 0))

        self.branch3x3dbl_1 = conv_block(in_channels, 448, kernel_size=1)
        self.branch3x3dbl_2 = conv_block(448, 384, kernel_size=3, padding=1)
        self.branch3x3dbl_3a = conv_block(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3dbl_3b = conv_block(384, 384, kernel_size=(3, 1), padding=(1, 0))

        self.branch_pool = conv_block(in_channels, 192, kernel_size=1)

    def _forward(self, x: Tensor) -> List[Tensor]:
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [
            self.branch3x3_2a(branch3x3),
            self.branch3x3_2b(branch3x3),
        ]
        branch3x3 = torch.cat(branch3x3, 1)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [
            self.branch3x3dbl_3a(branch3x3dbl),
            self.branch3x3dbl_3b(branch3x3dbl),
        ]
        branch3x3dbl = torch.cat(branch3x3dbl, 1)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return outputs

    def forward(self, x: Tensor) -> Tensor:
        outputs = self._forward(x)
        return torch.cat(outputs, 1)


class InceptionAux(nn.Module):

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        conv_block: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(InceptionAux, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.conv0 = conv_block(in_channels, 128, kernel_size=1)
        self.conv1 = conv_block(128, 768, kernel_size=5)
        self.conv1.stddev = 0.01  # type: ignore[assignment]
        self.fc = nn.Linear(768, num_classes)
        self.fc.stddev = 0.001  # type: ignore[assignment]

    def forward(self, x: Tensor) -> Tensor:
        # N x 768 x 17 x 17
        x = F.avg_pool2d(x, kernel_size=5, stride=3)
        # N x 768 x 5 x 5
        x = self.conv0(x)
        # N x 128 x 5 x 5
        x = self.conv1(x)
        # N x 768 x 1 x 1
        # Adaptive average pooling
        x = F.adaptive_avg_pool2d(x, (1, 1))
        # N x 768 x 1 x 1
        x = torch.flatten(x, 1)
        # N x 768
        x = self.fc(x)
        # N x 1000
        return x


class BasicConv2d(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        **kwargs: Any
    ) -> None:
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)

class PermInvariantClassifier(nn.Module):
    def __init__(self, n_classes = 3):
        super(PermInvariantClassifier, self).__init__()
        
        # (N, C, H, W) for pop1
        self.n1 = nn.Sequential(nn.Conv2d(1, 32, kernel_size = (1, 5), padding = (0, 0), stride = 1), nn.BatchNorm2d(32), nn.ReLU(),
                                nn.Conv2d(32, 64, kernel_size = (1, 5), padding = (0, 0), stride = 1), nn.BatchNorm2d(64), nn.ReLU())
        
        # (N, C, H, W) for pop2
        self.n2 = nn.Sequential(nn.Conv2d(1, 32, kernel_size = (1, 5), padding = (0, 0), stride = 1), nn.BatchNorm2d(32), nn.ReLU(),
                                nn.Conv2d(32, 64, kernel_size = (1, 5), padding = (0, 0), stride = 1), nn.BatchNorm2d(64), nn.ReLU())
        
        self.down = nn.MaxPool2d((1, 2))
        
        self.out = nn.Sequential(nn.Linear(7936, 4096), nn.BatchNorm1d(4096), nn.ReLU(), nn.Dropout(0.5),
                                 nn.Linear(4096, 4096), nn.BatchNorm1d(4096), nn.ReLU(), nn.Dropout(0.5),
                                 nn.Linear(4096, n_classes), nn.LogSoftmax(dim = -1))
        
        return
    
    def forward(self, x1, x2):
        x1 = self.down(self.n1(x1))
        x2 = self.down(self.n2(x2))
        
        x = torch.cat((x1, x2), dim = 2)
        x = torch.mean(x, dim = 2)
        
        x = torch.flatten(x, 1)
        #print(x.shape)
        
        x = self.out(x)
        
        return x
    
import torch
from torch import Tensor
import torch.nn as nn
from typing import Type, Any, Callable, Union, List, Optional


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-f37072fd.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-b627a593.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-0676ba61.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-63fe2227.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-394f9c45.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        in_channels: int = 2,
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(in_channels, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def _resnet(
    arch: str,
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    pretrained: bool,
    progress: bool,
    **kwargs: Any
) -> ResNet:
    model = ResNet(block, layers, **kwargs)
    return model


def resnet18(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)
    
class ResNetClassifier(nn.Module):
    def __init__(self):
        super(ResNetClassifier, self).__init__()
        
        self.net = nn.Sequential(*list(resnet18(pretrained = False).children())[:-2])
        
        self.out = nn.Sequential(nn.Linear(1024, 1024), nn.LayerNorm((1024, )), nn.ReLU(), 
                                 nn.Linear(1024, 1024), nn.LayerNorm((1024, )), nn.ReLU(), 
                                 nn.Linear(1024, 3), nn.LogSoftmax(dim = -1))
        
    def forward(self, x):
        x = self.net(x)
        print(x.shape)
        
        x = x.view(-1, 1024)
        
        x = self.out(x)
        
        return x
    
class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.act = nn.ELU()
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.InstanceNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.InstanceNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act(out)

        return out


class UNet(nn.Module):
    def __init__(self, num_classes, input_channels=3, **kwargs):
        super().__init__()

        nb_filter = [32, 64, 128, 256, 512]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])
        self.conv2_2 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv1_3 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv0_4 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)


    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))

        output = self.final(x0_4)
        return output
    
class NestedUNetIv3(nn.Module):
    def __init__(self, in_channels = 2):
        super().__init__()
        
        self.up1 = nn.Upsample(scale_factor = 2.098360655737705, mode = 'bilinear', align_corners = True)
        self.up2 = nn.Upsample(scale_factor = 2.18, mode = 'bilinear', align_corners = True)
        self.up3 = nn.Upsample(scale_factor = 2.1538461538461537, mode = 'bilinear', align_corners = True)
        
        m = Inception3(in_channels = 32)
        self.conv0_0 = VGGBlock(in_channels, 32, 32)
        # out: 32
        
        self.conv1_0 = nn.Sequential(*list(m.children())[:3])
        # out: 64
        
        self.conv2_0 = nn.Sequential(*list(m.children())[4:6])
        # out: 192
        
        self.conv3_0 = nn.Sequential(*list(m.children())[7:15])
        # out: 2048
        
        self.pool1 = list(m.children())[3]
        self.pool2 = list(m.children())[6]
        self.pool3 = list(m.children())[19]
        
        self.conv0_1 = VGGBlock(32 + 64, 32, 32)
        self.conv1_1 = VGGBlock(64 + 192, 64, 64)
        self.conv2_1 = VGGBlock(960, 192, 192)
        
        self.conv0_2 = VGGBlock(32 * 2 + 64, 32, 32)
        self.conv1_2 = VGGBlock(64 * 2 + 192, 64, 64)
        
        self.conv0_3 = VGGBlock(32 * 3 + 64, 32, 32)
        
        self.final = nn.Conv2d(32, 2, kernel_size = 1)
        
        self.out = nn.LogSigmoid()
    
    def forward(self, x):
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(x0_0)
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up1(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool1(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up2(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up1(x1_1)], 1))

        x3_0 = self.conv3_0(x2_0)
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up3(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up2(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up1(x1_2)], 1))
        
        output = self.final(x0_3)
        
        return output
    
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, k = 3, n_layers = 2, pooling = 'max'):
        super(ResBlock, self).__init__()

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for ix in range(n_layers):
            self.convs.append(nn.Conv2d(in_channels, out_channels, (k, k), 
                                        stride = (1, 1), padding = ((k + 1) // 2 - 1, (k + 1) // 2 - 1)))
            self.norms.append(nn.Sequential(nn.InstanceNorm2d(out_channels), nn.Dropout2d(0.1)))
            
            in_channels = out_channels
    
        self.activation = nn.ELU()
        
    def forward(self, x, return_unpooled = False):
        xs = [self.norms[0](self.convs[0](x))]
        
        for ix in range(1, len(self.norms)):
            xs.append(self.norms[ix](self.convs[ix](xs[-1])) + xs[-1])
            
        x = self.activation(torch.cat(xs, dim = 1))
        
        return x

# https://arxiv.org/pdf/1807.10165.pdf
class NestedUNet(nn.Module):
    def __init__(self, num_classes, input_channels=3, deep_supervision=False, **kwargs):
        super().__init__()

        nb_filter = [32, 64, 128, 256, 512]

        self.deep_supervision = deep_supervision

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor = 2, mode='bilinear', align_corners=True)

        self.dropout = nn.Dropout2d(0.2)

        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv0_1 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])

        self.conv0_2 = VGGBlock(nb_filter[0]*2+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = VGGBlock(nb_filter[1]*2+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_2 = VGGBlock(nb_filter[2]*2+nb_filter[3], nb_filter[2], nb_filter[2])

        self.conv0_3 = VGGBlock(nb_filter[0]*3+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_3 = VGGBlock(nb_filter[1]*3+nb_filter[2], nb_filter[1], nb_filter[1])

        self.conv0_4 = VGGBlock(nb_filter[0]*4+nb_filter[1], nb_filter[0], nb_filter[0])

        if self.deep_supervision:
            self.final1 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)


    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))
        x0_1 = self.dropout(x0_1)

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))
        x0_2 = self.dropout(x0_2)

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))
        x0_3 = self.dropout(x0_3)

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))
        x0_4 = self.dropout(x0_4)

        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            return [output1, output2, output3, output4]

        else:
            output = torch.squeeze(self.final(x0_4))
            return output

# in V2 I replace the VGG Block with Residual convolution blocks with the ELU activation function
class NestedUNetV2(nn.Module):
    def __init__(self, num_classes, input_channels=3, deep_supervision=False, **kwargs):
        super().__init__()

        nb_filter = [32, 64, 128, 256, 512]

        self.deep_supervision = deep_supervision

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor = 2, mode='bilinear', align_corners=True)

        self.dropout = nn.Dropout2d(0.1)

        self.conv0_0 = ResBlock(input_channels, nb_filter[0] // 2)
        self.conv1_0 = ResBlock(nb_filter[0], nb_filter[1] // 2)
        self.conv2_0 = ResBlock(nb_filter[1], nb_filter[2] // 2)
        self.conv3_0 = ResBlock(nb_filter[2], nb_filter[3] // 2)
        self.conv4_0 = ResBlock(nb_filter[3], nb_filter[4] // 2)

        self.conv0_1 = ResBlock(nb_filter[0]+nb_filter[1], nb_filter[0] // 2)
        self.conv1_1 = ResBlock(nb_filter[1]+nb_filter[2], nb_filter[1] // 2)
        self.conv2_1 = ResBlock(nb_filter[2]+nb_filter[3], nb_filter[2] // 2)
        self.conv3_1 = ResBlock(nb_filter[3]+nb_filter[4], nb_filter[3] // 2)

        self.conv0_2 = ResBlock(nb_filter[0]*2+nb_filter[1], nb_filter[0] // 2)
        self.conv1_2 = ResBlock(nb_filter[1]*2+nb_filter[2], nb_filter[1] // 2)
        self.conv2_2 = ResBlock(nb_filter[2]*2+nb_filter[3], nb_filter[2] // 2)

        self.conv0_3 = ResBlock(nb_filter[0]*3+nb_filter[1], nb_filter[0] // 2)
        self.conv1_3 = ResBlock(nb_filter[1]*3+nb_filter[2], nb_filter[1] // 2)

        self.conv0_4 = ResBlock(nb_filter[0]*4+nb_filter[1], nb_filter[0] // 2)

        if self.deep_supervision:
            self.final1 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)


    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            return [output1, output2, output3, output4]

        else:
            output = torch.squeeze(self.final(x0_4))
            return output



    
if __name__ == '__main__':
    model = NestedUNetIv3()
    
    x = torch.FloatTensor(np.zeros((4, 2, 128, 128)))
    print(model(x).shape)
    
    pass

