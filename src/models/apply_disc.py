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

from layers import NestedUNet, NestedUNetV2, LexStyleNet
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

    model = LexStyleNet()
    model = model.to(device)
        
    print(model)
    checkpoint = torch.load(args.weights, map_location = device)
    model.load_state_dict(checkpoint)
    
    model.eval()
    
    ifile = np.load(args.ifile)
    pop1_x = ifile['simMatrix'].T
    pop2_x = ifile['sechMatrix'].T
    
    X_ = np.vstack([pop1_x, pop2_x]).astype(np.uint8)
    ix = range(X_.shape[1])
    
    X = []
    P = []
    for c in chunks(ix, 508):
        X.append(X_[:,c])
        P.append([c[0], c[-1]])
        
    del X_
    X = np.array(X, dtype = np.uint8)
    
    Y = []
    for j in range(0, X.shape[0], 32):
        X_ = X[j:j + 32]
        
        x = torch.FloatTensor(X_).to(device)
        
        y_pred = model(x)
        y_pred = y_pred.detach().cpu().numpy()
        
        Y.extend(list(y_pred))

    np.savez(args.ofile, Y = np.array(Y), P = np.array(P))
    
if __name__ == '__main__':
    main()
