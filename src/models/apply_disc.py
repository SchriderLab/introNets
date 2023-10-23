# -*- coding: utf-8 -*-
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 16:21:13 2022

@author: kilgoretrout
"""

# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import logging
import argparse
import os

import torch

import h5py

import sys
sys.path.insert(0, os.path.join(os.getcwd(), 'src/models'))

from torchvision_mod_layers import resnet34
from data_loaders import H5UDataGenerator

from scipy.special import expit
import pickle

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser()
    # my args
    parser.add_argument("--verbose", action = "store_true", help = "display messages")
    
    parser.add_argument("--weights", default = "None") # weights of the pre-trained model
    parser.add_argument("--ifile", default = "None")
    parser.add_argument("--in_channels", default = "2")
    parser.add_argument("--n_classes", default = "4")
    
    parser.add_argument("--ofile", default = "test.npz")
    
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

    logging.info('creating model...')
    # ${code_blocks}
    device = torch.device('cuda')

    model = resnet34(in_channels = int(args.in_channels), num_classes = int(args.n_classes))
    model = model.to(device)
        
    print(model)
    checkpoint = torch.load(args.weights, map_location = device)
    model.load_state_dict(checkpoint)
    
    model.eval()
    
    ifile = h5py.File(args.ifile, 'r')
    keys = list(ifile.keys())
    
    shape = tuple(ifile['shape'])
    l = shape[-1]
    
    Y = np.zeros((l, int(args.n_classes)))
    count = np.zeros((l, int(args.n_classes)))
        
    for key in keys:
        try:
            int(key)
        except:
            continue
        
        logging.info('working on key {}...'.format(key))
        
        X = np.array(ifile[key]['x_0'])
        pos = np.array(ifile[key]['positions'])
        indices_ = np.array(ifile[key]['pi'])
        
        # let's forward the whole batch through segmentation
        with torch.no_grad():
            x = torch.FloatTensor(X).to(device)

            y_pred = model(x)
        
        y_pred = y_pred.detach().cpu().numpy()
        
        for k in range(indices_.shape[0]):
            ip = indices_[k]
            
            Y[ip,:] += y_pred[k]
            count[ip,:] += 1
            
    Y = expit(Y / count)
            
    np.savez(args.ofile, Y = Y)
    
if __name__ == '__main__':
    main()
