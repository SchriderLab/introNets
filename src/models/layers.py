import torch.nn as nn
import numpy as np
from collections import defaultdict

from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d as BN, Dropout
from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

from torch import autograd
from typing import Callable, Any, Optional, Tuple, List

import warnings
from torch import Tensor


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
        
        self.down = nn.AvgPool1d(2)
        
        in_channels = h
        out_channels = [256, 128, 128]
        for ix in range(n_layers):
            self.convs.append(nn.Sequential(nn.Conv1d(in_channels, out_channels[ix], 2), nn.InstanceNorm1d(out_channels[ix]), nn.ReLU(), nn.Dropout(0.25)))
            
            in_channels = copy.copy(out_channels[ix])
            
            w = w // 2
        
        features = 3
        
        self.out_size = 8064
        self.out = nn.Sequential(nn.Linear(128, 128), nn.LayerNorm((128,)), nn.ReLU(),
                                 nn.Linear(128, 3), nn.Softmax(dim = -1))
    def forward(self, x):
        for ix in range(len(self.convs)):
            x = self.convs[ix](x)
            x = self.down(x)
        
        x = x.mean(dim = -1)
        x = x.view(-1, 128)
        
        return self.out(x)
        
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
    
from torch import Tensor
import torch.nn as nn

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

# in V2 I replace the VGG Block with Residual convolution blocks with the ELU activation function
class NestedUNet(nn.Module):
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


