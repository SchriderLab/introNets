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

from layers import NestedUNet, NestedUNetV2
from data_loaders import H5UDataGenerator

from scipy.special import expit
import pickle

import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix
import pandas as pd

def cm_analysis(y_true, y_pred, filename, labels, ymap=None, figsize=(10,10)):
    """
    Generate matrix plot of confusion matrix with pretty annotations.
    The plot image is saved to disk.
    args: 
      y_true:    true label of the data, with shape (nsamples,)
      y_pred:    prediction of the data, with shape (nsamples,)
      filename:  filename of figure file to save
      labels:    string array, name the order of class labels in the confusion matrix.
                 use `clf.classes_` if using scikit-learn models.
                 with shape (nclass,).
      ymap:      dict: any -> string, length == nclass.
                 if not None, map the labels & ys to more understandable strings.
                 Caution: original y_true, y_pred and labels must align.
      figsize:   the size of the figure plotted.
    """
    if ymap is not None:
        y_pred = [ymap[yi] for yi in y_pred]
        y_true = [ymap[yi] for yi in y_true]
        labels = [ymap[yi] for yi in labels]
    cm = confusion_matrix(y_true, y_pred)
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
            elif c == 0:
                annot[i, j] = ''
            else:
                annot[i, j] = '%.1f%%\n%d' % (p, c)
    cm = pd.DataFrame(cm, index=labels, columns=labels)
    cm.index.name = 'Actual'
    cm.columns.name = 'Predicted'
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm, annot=annot, fmt='', ax=ax)
    plt.savefig(filename)
    plt.close()


def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser()
    # my args
    parser.add_argument("--verbose", action = "store_true", help = "display messages")
    
    parser.add_argument("--weights", default = "None") # weights of the pre-trained model
    parser.add_argument("--disc", default = "None")
    
    parser.add_argument("--ifile", default = "None")
    
    parser.add_argument("--keys", default = "None")
    parser.add_argument("--odir", default = "None")
    parser.add_argument("--n_samples", default = "100")
    parser.add_argument("--ofile", default = "test.npz")
    parser.add_argument("--sigma", default = "5")
    parser.add_argument("--n", default = "128")
    
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

from scipy.ndimage import gaussian_filter1d

def main():
    args = parse_args()
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    model = NestedUNetV2(1, 2)
    if args.weights != "None":
        checkpoint = torch.load(args.weights, map_location = device)
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()
    
    ifile = h5py.File(args.ifile, 'r')
    keys = list(ifile.keys())
    
    shape = tuple(ifile['shape'])
    print(shape)
    l = shape[-1]
    
    Y = np.zeros((32, l), dtype = np.float32)
    count = np.zeros((1, l), dtype = np.float32)
        
    x1_indices = np.array(ifile['x1_indices'])
    x2_indices = np.array(ifile['x2_indices'])
    
    G = gaussian(int(args.n), int(args.sigma))
    Gn = G.detach().cpu().numpy()
    
    plt.plot(Gn)
    plt.savefig('gauss_view.png', dpi = 100)
    plt.close()
    
    G = G.view(1, 1, int(args.n)).to(device)
    
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

            y_pred = torch.squeeze(model(x))
            
            # platt scale and apply Gaussian
            y_pred = (y_pred * 1.1138 - 1.3514) * G
            
            y_pred = y_pred.detach().cpu().numpy()
            
            x = x.detach().cpu().numpy()
            
            # add the predictions to the overall genome-wide prediction
            for k in range(y_pred.shape[0]):
                ip = indices_[k]
                
                #print(ip)
                
                i1 = list(indices[k][0])
                i2 = list(indices[k][1])
                
                #print(i2, np.argsort(i2))
                
                i2 = np.argsort(i2)
                
                # reorder the matrix
                y_pred[k,:,:] = y_pred[k,i2,:]
                #y_pred = gaussian_filter1d(y_pred, 1)
            
                """
                fig, axes = plt.subplots(nrows = 3)
                axes[0].imshow(x[k,0,:,:])
                axes[1].imshow(x[k,1,:,:])
                im = axes[2].imshow(expit(y_pred[k]), vmin = 0., vmax = 1.)
                
                fig.colorbar(im, ax = axes[2])
                plt.show()
                """
                
                Y[:,ip] += y_pred[k,:,:]
                count[:,ip] += Gn.reshape(1, int(args.n))
    
    ix = list(np.where(np.sum(count, axis = 0) >= 1)[0])
    Y = Y[:, ix] / count[:, ix]
    
    np.savez(args.ofile, Y = expit(Y), x1i = x1_indices, x2i = x2_indices)
                
            
    
if __name__ == '__main__':
    main()