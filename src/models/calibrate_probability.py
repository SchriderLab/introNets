# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-

import os
import argparse
import logging

import torch
from torch.nn import CrossEntropyLoss, KLDivLoss, NLLLoss, SmoothL1Loss, BCELoss, BCEWithLogitsLoss
import torch

from torch import nn

import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.utils.data
import torch.utils.data.distributed
import torch.nn.functional as F
import torch.distributed as dist
import copy
from scipy.special import expit

from layers import NestedUNet
from data_loaders import H5UDataGenerator
import h5py

import numpy as np
from sklearn.metrics import accuracy_score
import pandas as pd
import pickle
# use this format to tell the parsers
# where to insert certain parts of the script
# ${imports}

import matplotlib
matplotlib.use('Agg')

import torch.nn.functional as F

import matplotlib.pyplot as plt

class ProbCalModel(nn.Module):
    def __init__(self, model):
        super(ProbCalModel, self).__init__()
        
        self.model = model
        
        for p in self.model.parameters():
            p.requires_grad = False
        
        self.a = torch.nn.parameter.Parameter(torch.ones((1,)), requires_grad = True)
        self.b = torch.nn.parameter.Parameter(torch.zeros((1,)), requires_grad = True)
        
        self.sig = nn.Sigmoid()
        
    def forward(self, x):
        x = self.model(x)
        
        x = x * self.a + self.b
        
        return x

def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser()
    # my args
    parser.add_argument("--verbose", action = "store_true", help = "display messages")
    parser.add_argument("--weights", default = "data/training_results_i4/i2_fcn_nested_res_BA_n128.weights") # weights of the pre-trained model
    parser.add_argument("--ifile", default = "data/BA_seg_n128.hdf5")
    parser.add_argument("--keys", default = "data/training_results_i4/i2_fcn_nested_res_BA_n128_val_keys.pkl")
    
    parser.add_argument("--ofile", default = "None")
    parser.add_argument("--n_samples", default = "100")
    
    parser.add_argument("--n_epochs", default = "100")
    parser.add_argument("--out_channels", default = "2")
    
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
        logging.debug("running in verbose mode")
    else:
        logging.basicConfig(level=logging.INFO)

    return args

def main():
    args = parse_args()
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    model = NestedUNet(int(args.out_channels), 2)
    if args.weights != "None":
        checkpoint = torch.load(args.weights, map_location = device)
        model.load_state_dict(checkpoint)

    generator = H5UDataGenerator(h5py.File(args.ifile, 'r'), batch_size = 32)
    generator.val_keys = pickle.load(open(args.keys, 'rb'))

    generator.define_lengths()

    model = ProbCalModel(model)
    model = model.to(device)
    criterion = BCEWithLogitsLoss()
    
    optimizer = optim.Adam(model.parameters(), lr = 0.001)
    
    min_loss = np.inf
    
    for ij in range(int(args.n_epochs)):
        losses = []
        
        for ix in range(generator.val_length):
            optimizer.zero_grad()
            x, y = generator.get_val_batch()
            
            x = x.to(device)
            y = y.to(device)
    
            y_pred = model(x)
            
            loss = criterion(y_pred, torch.squeeze(y)) # ${loss_change}
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
            
        generator.on_epoch_end()
            
        if np.mean(losses) < min_loss:
            current = (model.a.item(), model.b.item())
            min_loss = copy.copy(np.mean(losses))
            print('updating weights...')
        
        print('have loss of {}...'.format(np.mean(losses)))
    
    np.savetxt(args.ofile, np.array(current))
    
if __name__ == '__main__':
    main()