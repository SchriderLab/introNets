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
    parser.add_argument("--ifile", default = "None")
    parser.add_argument("--keys", default = "None")
    
    parser.add_argument("--odir", default = "None")
    parser.add_argument("--n_samples", default = "100")
    
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
    
    generator = H5UDataGenerator(h5py.File(args.ifile, 'r'), batch_size = 4)
    generator.val_keys = pickle.load(open(args.keys, 'rb'))
    
    counter = 0
    
    Y = []
    Y_pred = []
    for ix in range(int(args.n_samples)):
        with torch.no_grad():
            x, y = generator.get_val_batch()
            
            x = x.to(device)
            y = y.to(device)

            y_pred = model(x)
            
            x = x.detach().cpu().numpy()
            y = y.detach().cpu().numpy()
            y_pred = y_pred.detach().cpu().numpy()
            
            print(x.shape, y.shape, y_pred.shape)
            
            Y.extend(y.flatten())
            Y_pred.extend(expit(y_pred.flatten()))
            
            for k in range(x.shape[0]):
                
                fig = plt.figure(figsize=(16, 6))
                ax0 = fig.add_subplot(151)
                
                ax0.imshow(x[k,0,:,:], cmap = 'gray')
                ax0.set_title('pop A')
                
                ax1 = fig.add_subplot(152)
                ax1.imshow(x[k,1,:,:], cmap = 'gray')
                ax1.set_title('pop B')
                
                ax2 = fig.add_subplot(153)
                ax2.imshow(y[k,0,:,:], cmap = 'gray')
                ax2.set_title('pop B (y)')
                
                ax3 = fig.add_subplot(154)
                ax3.imshow(np.round(expit(y_pred[k,:,:])), cmap = 'gray')
                ax3.set_title('pop B (pred)')
                
                ax4 = fig.add_subplot(155)
                im = ax4.imshow(expit(y_pred[k,:,:]))
                fig.colorbar(im, ax = ax4)
                
                plt.savefig(os.path.join(args.odir, '{0:04d}_pred.png'.format(counter)), dpi = 100)
                counter += 1
                plt.close()
                
    cm_analysis(Y, np.round(Y_pred), os.path.join(args.odir, 'confusion_matrix.png'), ['native', 'introgressed'])

    
if __name__ == '__main__':
    main()