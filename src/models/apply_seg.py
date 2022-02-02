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
    count = np.zeros((32, l), dtype = np.float32)
    
    x1_indices = np.array(ifile['x1_indices'])
    x2_indices = np.array(ifile['x2_indices'])
    
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
            y_pred = expit(y_pred.detach().cpu().numpy())
            
            # add the predictions to the overall genome-wide prediction
            for k in range(y_pred.shape[0]):
                ip = indices_[k]
                i1 = list(indices[k][0])
                i2 = list(indices[k][1])
                
                # reorder the matrices
                y_pred[k,:,:] = y_pred[k,i2,:]

                Y[:,ip] += y_pred[k,:,:]
                count[:,ip] += 1
    
    ix = list(np.where(np.sum(count, axis = 0) != 0)[0])
    Y = Y[:, ix] / count[:, ix]
    
    np.savez(args.ofile, Y = Y, x1i = x1_indices, x2i = x2_indices)
                
            
    
if __name__ == '__main__':
    main()