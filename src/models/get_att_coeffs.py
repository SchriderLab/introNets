# -*- coding: utf-8 -*-
import os
import argparse
import logging
import sys

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

from layers import NestedUNet
from data_loaders import H5UDataGenerator
import h5py

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import pandas as pd
# use this format to tell the parsers
# where to insert certain parts of the script
# ${imports}

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from layers import GCNUNet_i2
from data_loaders import GCNDataGenerator

from torch_scatter import scatter_max, scatter_mean, scatter_std
from scipy.special import expit

import glob
from train_gcn_unet_transfer import TransferModel

# use this format to tell the parsers
# where to insert certain parts of the script
# ${imports}

def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser()
    # my args
    parser.add_argument("--verbose", action = "store_true", help = "display messages")
    parser.add_argument("--weights", default = "None")
    
    parser.add_argument("--batch_size", default = "16")
    parser.add_argument("--n_features", default = "128")
    parser.add_argument("--n_global", default = "512")
    parser.add_argument("--n_heads", default = "2")
    parser.add_argument("--layer_type", default = "gat")
    
    parser.add_argument("--idir", default = "None")
    parser.add_argument("--n_samples", default = "100")

    parser.add_argument("--odir", default = "None")
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

def main():
    args = parse_args()
    
    device = torch.device('cuda')
        
    ifile = glob.glob(os.path.join(args.idir, '*/*.npz'))[0]
    ifile = np.load(ifile, allow_pickle = True)
    
    n_layers = len(ifile['edges'])
    
    model = GCNUNet_i2(in_channels = 306, n_features = int(args.n_features), 
                    n_classes = 1, layer_type = args.layer_type, n_layers = n_layers, 
                    n_heads = int(args.n_heads), n_global = int(args.n_global), return_attention_weights = True)       
    
    model = TransferModel(model)
    
    checkpoint = torch.load(args.weights, map_location = device)
    model.load_state_dict(checkpoint)
    model = model.to(device)
    
    generator = GCNDataGenerator(args.idir, batch_size = int(args.batch_size), seg = True)
    
    d = []
    alpha = []
    
    model.eval()
    for ix in range(int(args.n_samples)):
        x, y, edges = generator.get_element()
        
        x = x.to(device)
        y = y.to(device)
        
        edges = [u.to(device) for u in edges]
        batch = torch.LongTensor(np.zeros(x.shape[0])).to(device)
        
        with torch.no_grad():
            y_pred, att_weights, att_edges = model.forward(x, edges, batch, return_attention_weights = True)
            
    
        for ix in range(len(att_weights)):
            e = att_edges[ix].detach().cpu().numpy()
            a = np.sum(np.abs(att_weights[ix].detach().cpu().numpy()), axis = 1)

            e = np.abs(e[0] - e[1])
            
            print(e.shape, a.shape)

            d.extend(e)
            alpha.extend(a)
            
    plt.scatter(d, alpha, alpha = 0.01)
    plt.xlabel('distance (n snps away)')
    plt.ylabel('summed absolute attention coeff')
    
    plt.savefig('att_weights.png', dpi = 100)
    plt.close()
    
if __name__ == '__main__':
    main()

