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

from layers import PermInvariantClassifier
from data_loaders import H5DisDataGenerator
import h5py

import numpy as np
from sklearn.metrics import accuracy_score
import pandas as pd

from data_functions import load_data_dros

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser()
    # my args
    parser.add_argument("--verbose", action = "store_true", help = "display messages")
    parser.add_argument("--weights", default = "None")
    
    #parser.add_argument("--ifile", default = "None", help = "initial random simulation data the discriminator was trained on")
    parser.add_argument("--idir", default = "None")
    parser.add_argument("--npzs", default = "None")
    
    parser.add_argument("--n_steps", default = "100")
    
    parser.add_argument("--odir", default = "None")
    parser.add_argument("--T", default = "1.0")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
        logging.debug("running in verbose mode")
    else:
        logging.basicConfig(level=logging.INFO)

    if args.odir != "None":
        if not os.path.exists(args.odir):
            os.system('mkdir -p {}'.format(args.odir))
            logging.debug('root: made output directory {0}'.format(args.odir))
    # ${odir_del_block}

    return args

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def main():
    args = parse_args()
    device = torch.device('cuda')

    model = PermInvariantClassifier()
    model = model.to(device)
    
    checkpoint = torch.load(args.weights, map_location = device)
    model.load_state_dict(checkpoint)
    
    model.eval()
    
    idirs = [os.path.join(args.idir, u) for u in os.listdir(args.idir)]
    
    P = []
    l = []
    for idir in idirs:
        print('predicting on {}...'.format(idir))
        
        ms = os.path.join(idir, 'mig.msOut')
        anc = os.path.join(idir, 'out.anc')
        
        x1, x2, y1, y2, params = load_data_dros(ms, anc)
        x1 = torch.FloatTensor(np.expand_dims(x1, axis = 1))
        x2 = torch.FloatTensor(np.expand_dims(x2, axis = 1))
        
        # theta, theta_rho, nu_ab, nu_ba, alpha1, alpha2, T, migTime, migProb
        p = params[0,[0, 1, 2, 3, 4, 5, 8, 10, 11]]
        P.append(p)
        
        ys = []
        for c in chunks(list(range(x1.shape[0])), 50):
            x1_ = x1[c,::].to(device)
            x2_ = x2[c,::].to(device)
            
            with torch.no_grad():
                y_pred = model(x1_, x2_).detach().cpu().numpy()
                
                # log probability of real classificiation
                y_ = -y_pred[:,1]
                
                ys.extend(list(y_))
                
        print('got nll of: {}...'.format(np.mean(ys)))
        l.append(np.mean(ys))
        
    plt.plot(sorted(l))
    plt.savefig('nll.png', dpi = 100)
    plt.close()
    
if __name__ == '__main__':
    main()
    
