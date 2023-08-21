# -*- coding: utf-8 -*-
import matplotlib
#matplotlib.use('Agg')

import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 18})


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
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score, confusion_matrix, accuracy_score
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
                annot[i, j] = '%.1f%%' % (p)
            elif c == 0:
                annot[i, j] = ''
            else:
                annot[i, j] = '%.1f%%' % (p)
    cm = pd.DataFrame(cm_perc, index=labels, columns=labels)
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
    
    parser.add_argument("--odir", default = "None")
    parser.add_argument("--n_classes", default = "1")
    parser.add_argument("--n_samples", default = "250")
    parser.add_argument("--n_to_plot", default = "25")
    
    parser.add_argument("--plot_both", action = "store_true")
    parser.add_argument("--plot_label", default = "2")
    
    parser.add_argument("--platt_scaling", default = "None")
    
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

    model = NestedUNet(int(args.n_classes), 2)
    if args.weights != "None":
        checkpoint = torch.load(args.weights, map_location = device)
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()
    
    if args.platt_scaling != "None":
        a, b = np.loadtxt(args.platt_scaling)
        ab_platt = (torch.FloatTensor([a]).to(device), 
                    torch.FloatTensor([b]).to(device))
    else:
        ab_platt = None
    
    generator = H5UDataGenerator(h5py.File(args.ifile, 'r'), batch_size = 4, val_prop = 0, label_smooth = False)
    
    print(generator.length)
    if args.n_samples == "None":
        N = generator.length
    else:
        N = int(args.n_samples)
    
    if N > generator.length:
        N = generator.length
    
    counter = 0
    
    Y = []
    Y_pred = []
    
    print('predicting... on {} mini-batches...'.format(N))
    for ix in range(N):
        with torch.no_grad():
            x, y = generator.get_batch()
            
            if len(x.shape) > 4:
                x = torch.squeeze(x)
                y = torch.squeeze(y)
            
            if y is None:
                break

            x = x.to(device)
            y = y.to(device)
            
            print(x.shape, y.sahpe)

            y_pred = model(x)
            
            if ab_platt is not None:
                a, b = ab_platt
                y_pred = y_pred * a + b
            
            x = x.detach().cpu().numpy()
            y = y.detach().cpu().numpy()
            y_pred = y_pred.detach().cpu().numpy()
            
            Y.extend(y.flatten())
            Y_pred.extend(expit(y_pred.flatten()))
            
            if counter < int(args.n_to_plot):
                k = 0
                if not args.plot_both:
                    fig = plt.figure(figsize=(16, 6))
                    ax0 = fig.add_subplot(151)
                    
                    ax0.imshow(x[k,0,:,:], cmap = 'gray')
                    ax0.set_title('pop 1')
                    
                    ax1 = fig.add_subplot(152)
                    ax1.imshow(x[k,1,:,:], cmap = 'gray')
                    ax1.set_title('pop 2')
                    
                    ax2 = fig.add_subplot(153)
                    ax2.imshow(y[k,0,:,:], vmin = 0., vmax = 1., cmap = 'gray')
                    ax2.set_title('pop {} (y)'.format(args.plot_label))
                    
                    ax3 = fig.add_subplot(154)
                    ax3.imshow(np.round(expit(y_pred[k,:,:])), vmin = 0., vmax = 1., cmap = 'gray')
                    ax3.set_title('pop {} (pred)'.format(args.plot_label))
                    
                    ax4 = fig.add_subplot(155)
                    im = ax4.imshow(expit(y_pred[k,:,:]))
                    fig.colorbar(im, ax = ax4)
                    ax4.set_title('prob')
                    
                    plt.savefig(os.path.join(args.odir, '{0:04d}_pred.eps'.format(counter)), dpi = 100)
                else:
                    fig = plt.figure(figsize=(16, 6))
                    
                    ax = fig.add_subplot(241)
                    ax.imshow(x[k,0,:,:], cmap = 'gray')
                    ax.set_title('pop 1')
                    
                    ax = fig.add_subplot(242)
                    ax.imshow(y[k,0,:,:], vmin = 0., vmax = 1., cmap = 'gray')
                    ax.set_title('pop 1 (y)')
                    
                    ax = fig.add_subplot(243)
                    ax.imshow(np.round(expit(y_pred[k,0,:,:])), vmin = 0., vmax = 1., cmap = 'gray')
                    ax.set_title('pop 1 (pred)')
                    
                    ax = fig.add_subplot(244)
                    im = ax.imshow(expit(y_pred[k,0,:,:]))
                    fig.colorbar(im, ax = ax)
                    ax.set_title('prob')
                    
                    ax = fig.add_subplot(245)
                    ax.imshow(x[k,1,:,:], cmap = 'gray')
                    ax.set_title('pop 2')
                    
                    ax = fig.add_subplot(246)
                    ax.imshow(y[k,1,:,:], vmin = 0., vmax = 1., cmap = 'gray')
                    ax.set_title('pop 2 (y)')
                    
                    ax = fig.add_subplot(247)
                    ax.imshow(np.round(expit(y_pred[k,1,:,:])), vmin = 0., vmax = 1., cmap = 'gray')
                    ax.set_title('pop 2 (pred)')
                    
                    ax = fig.add_subplot(248)
                    im = ax.imshow(expit(y_pred[k,1,:,:]))
                    fig.colorbar(im, ax = ax)
                    ax.set_title('prob')
                    
                    plt.savefig(os.path.join(args.odir, '{0:04d}_pred.eps'.format(counter)), dpi = 100)
                
                counter += 1
                plt.close()
            
            
    print(len(Y))
    print('plotting...')
    # what probability bin do they fall in?
    p_bins = np.linspace(0., 1., 15)
    p = np.diff(p_bins) / 2. + p_bins[:-1]
    
    # count whether or not we find a positive label here
    count_pos = np.zeros(len(p_bins) - 1, dtype = np.float32)
    count_neg = np.zeros(len(p_bins) - 1, dtype = np.float32)
    
    Y = np.array(Y)
    Y_pred = np.array(Y_pred)
    
    Y_pred_bin = np.digitize(Y_pred, p_bins)
    for k in range(len(Y)):
        if Y[k] == 0:
            count_neg[min([Y_pred_bin[k] - 1, count_neg.shape[0] - 1])] += 1
        else:
            count_pos[min([Y_pred_bin[k] - 1, count_neg.shape[0] - 1])] += 1
    
    count = count_pos / (count_neg + count_pos)
    
    plt.scatter(p, count)
    plt.plot([0, 1], [0, 1])
    plt.xlabel('predicted probability')
    plt.ylabel('acccuracy')
    plt.savefig(os.path.join(args.odir, 'probability_calibaration.eps'))
    plt.close()
    
    Y = np.round(np.array(Y))
    
    cm_analysis(Y, np.round(Y_pred), os.path.join(args.odir, 'confusion_matrix.eps'), ['not introgressed', 'introgressed'])
    
    precision, recall, thresholds = precision_recall_curve(list(map(int, Y)), Y_pred)
    fpr, tpr, _ = roc_curve(list(map(int, Y)), Y_pred)
    
    fig = plt.figure(figsize=(12, 12))
    ax0 = fig.add_subplot(111)
    ax0.plot(fpr, tpr)
    ax0.set_xlabel('false positive rate')
    ax0.set_ylabel('true positiive rate')
    
    plt.savefig(os.path.join(args.odir, 'roc_curve.eps'))
    plt.close()
    
    fig = plt.figure(figsize=(12, 12))
    ax0 = fig.add_subplot(111)
    ax0.plot(recall, precision)
    ax0.set_xlabel('recall')
    ax0.set_ylabel('precision')
    plt.savefig(os.path.join(args.odir, 'precision_recall.eps'))
    plt.close()
    
    auroc = roc_auc_score(list(map(int, Y)), Y_pred)
    aupr = average_precision_score(list(map(int, Y)), Y_pred)
    
    ofile = open(os.path.join(args.odir, 'metrics.txt'), 'w')
    
    print('auroc: {}'.format(auroc), file = ofile)
    print('aupr: {}'.format(aupr), file = ofile)
    print('accuracy: {}'.format(accuracy_score(list(map(int, Y)), np.round(Y_pred))), file = ofile)

if __name__ == '__main__':
    main()