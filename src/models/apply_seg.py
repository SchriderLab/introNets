#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 16:21:13 2022

@author: kilgoretrout
"""

from typing import Tuple

import torch
import torch.nn as nn
from torch.nn.functional import conv2d

import matplotlib
matplotlib.use('Agg')

def gaussian(window_size, sigma):
    def gauss_fcn(x):
        return -(x - window_size // 2)**2 / float(2 * sigma**2)
    gauss = torch.stack(
        [torch.exp(torch.tensor(gauss_fcn(x))) for x in range(window_size)])
    return gauss / gauss.max()


def get_gaussian_kernel(ksize: int, sigma: float) -> torch.Tensor:
    r"""Function that returns Gaussian filter coefficients.

    Args:
        ksize (int): filter size. It should be odd and positive.
        sigma (float): gaussian standard deviation.

    Returns:
        Tensor: 1D tensor with gaussian filter coefficients.

    Shape:
        - Output: :math:`(ksize,)`

    Examples::

        >>> tgm.image.get_gaussian_kernel(3, 2.5)
        tensor([0.3243, 0.3513, 0.3243])

        >>> tgm.image.get_gaussian_kernel(5, 1.5)
        tensor([0.1201, 0.2339, 0.2921, 0.2339, 0.1201])
    """
    if not isinstance(ksize, int) or ksize % 2 == 0 or ksize <= 0:
        raise TypeError("ksize must be an odd positive integer. Got {}"
                        .format(ksize))
    window_1d: torch.Tensor = gaussian(ksize, sigma)
    
    return window_1d

# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import logging
import argparse
import os

import torch

import h5py

import sys
sys.path.insert(0, os.path.join(os.getcwd(), 'src/models'))

from layers import NestedUNet
from data_loaders import H5UDataGenerator

from scipy.special import expit
import pickle

import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix
import pandas as pd

def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser()
    # my args
    parser.add_argument("--verbose", action = "store_true", help = "display messages")
    
    parser.add_argument("--weights", default = "None") # weights of the pre-trained model
    parser.add_argument("--ifile", default = "None")

    parser.add_argument("--platt", default = "None")
    parser.add_argument("--pop", default = "1", help = "population that were predicting on.  0, 1 or -1 for both")
    
    parser.add_argument("--smooth", action = "store_true")    
    parser.add_argument("--sigma", default = "5")
    parser.add_argument("--n", default = "128")
    
    parser.add_argument("--keys", default = "simMatrix,sechMatrix")
    parser.add_argument("--odir", default = "None")
    parser.add_argument("--n_samples", default = "100")
    parser.add_argument("--ofile", default = "test.npz")
    
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
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    model = NestedUNet(1, 2)
    if args.weights != "None":
        checkpoint = torch.load(args.weights, map_location = device)
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()
    
    ifile = h5py.File(args.ifile, 'r')
    keys = list(ifile.keys())

    shape = tuple(ifile['shape'])
    l = shape[-1]
    
    pop = int(args.pop)
    
    # are we predicting on a single pop or both?
    if pop in [0, 1]:
        Y = np.zeros((ifile['0']['x_0'].shape[2], 1, l), dtype = np.float32)
    else:
        Y = np.zeros((ifile['0']['x_0'].shape[2], 2, l), dtype = np.float32)
    
    count = np.zeros((1, 1, l), dtype = np.float32)
        
    x1_indices = np.array(ifile['x1_indices'])
    x2_indices = np.array(ifile['x2_indices'])

    # if we have platt coefficients for confidence correction    
    if args.platt != "None":
        platt = tuple(np.loadtxt(args.platt))
    else:
        platt = None
    
    G = gaussian(int(args.n), int(args.sigma))
    Gn = G.detach().cpu().numpy()
        
    G = G.view(1, 1, 1, int(args.n)).to(device)
    
    for key in keys:
        try:
            int(key)
        except:
            continue
        
        logging.info('working on key {}...'.format(key))
        
        X = np.array(ifile[key]['x_0'])
        indices = np.array(ifile[key]['indices'])
        indices_ = np.array(ifile[key]['pi'])
        pos = np.array(ifile[key]['positions'])

        # let's forward the whole batch through segmentation
        with torch.no_grad():
            x = torch.FloatTensor(X).to(device)

            y_pred = model(x)
            
            # platt scale and apply Gaussian (if specified)
            if platt is not None:
                y_pred = (y_pred * platt[0] + platt[1])
                
            if args.smooth:
                y_pred *= G
            
            y_pred = y_pred.detach().cpu().numpy()
            x = x.detach().cpu().numpy()
            
            if len(y_pred.shape) == 3:
                y_pred = np.expand_dims(y_pred, 1)
            
            # add the predictions to the overall genome-wide prediction
            for k in range(y_pred.shape[0]):
                ip = indices_[k]
                
                i1 = list(indices[k][0])
                i2 = list(indices[k][1])
                
                i1 = np.argsort(i1)
                i2 = np.argsort(i2)
                
                if pop == 1:
                    # reorder the matrix
                    y_pred[k,0,:,:] = y_pred[k,0,i2,:]
                elif pop == 0:
                    y_pred[k,0,:,:] = y_pred[k,0,i1,:]
                else:
                    y_pred[k,0,:,:] = y_pred[k,0,i1,:]
                    y_pred[k,1,:,:] = y_pred[k,1,i2,:]
            
                if args.smooth:
                    Y[:,:,ip] += y_pred[k,:,:,:].transpose(1,0,2)
                    count[:,:,ip] += Gn.reshape(1, 1, int(args.n))
                else:
                    Y[:,:,ip] += y_pred[k,:,:,:].transpose(1,0,2)
                    count[:,:,ip] += 1.

    Y = Y / count
    
    np.savez(args.ofile, Y = expit(Y), x1i = x1_indices, x2i = x2_indices)
                
            
    
if __name__ == '__main__':
    main()