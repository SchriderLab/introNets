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

from train_discriminator import cm_analysis

def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser()
    # my args
    parser.add_argument("--verbose", action = "store_true", help = "display messages")
    parser.add_argument("--weights", default = "None", help = "weights of the pretrained model") # weights of the pre-trained model
    parser.add_argument("--ifile", default = "None", help = "data to predict on")
    
    parser.add_argument("--odir", default = "None")
    parser.add_argument("--n_classes", default = "1", help = "number of output channels in predicted image, or the number of populations to predict on.  Has to match the weights and input data")
    parser.add_argument("--n_samples", default = "None", help = "number of random keys in the h5 file to predict.  if left None then use them all")
    parser.add_argument("--n_to_plot", default = "25", help = "plots some random sample of the x, y, and predicted y variables")

    parser.add_argument("--plot_label", default = "2")
    parser.add_argument("--keys", default = "None")
    
    parser.add_argument("--format", default = "png")
    
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
        else:
            os.system('rm -rf {}'.format(os.path.join(args.odir, '*')))

    return args

def main():
    args = parse_args()
    
    if int(args.n_classes) == 2:
        args.plot_both = True
    else:
        args.plot_both = False
    
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
    if args.keys != "None":
        keys = pickle.load(open(args.keys, 'rb'))
    
        generator.keys = keys
    
    if args.n_samples == "None":
        N = generator.length
    else:
        N = int(args.n_samples)
    
    if N > generator.length:
        N = generator.length
    
    counter = 0
    
    Y = []
    Y_pred = []
    
    accs = []
    
    logging.info('predicting... on {} mini-batches...'.format(N))
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

            y_pred = model(x)
            
            if ab_platt is not None:
                a, b = ab_platt
                y_pred = y_pred * a + b
            
            x = x.detach().cpu().numpy()
            y = y.detach().cpu().numpy()
            y_pred = y_pred.detach().cpu().numpy()
            
            accs.append(accuracy_score(y.flatten(), np.round(expit(y_pred.flatten()))))
            
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
                    
                    plt.savefig(os.path.join(args.odir, '{0:04d}_pred.{1}'.format(counter, args.format)), dpi = 100)
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
                    
                    plt.savefig(os.path.join(args.odir, '{0:04d}_pred.{1}'.format(counter, args.format)), dpi = 100)
                
                counter += 1
                plt.close()
            
          
    logging.info('got mean pixel accuracy of {}...'.format(np.mean(accs)))  
          
    logging.info('plotting metrics for {} pixels...'.format(len(Y)))
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
    
    acc = accuracy_score(list(map(int, Y)), np.round(Y_pred))
    
    print('auroc: {}'.format(auroc), file = ofile)
    print('aupr: {}'.format(aupr), file = ofile)
    print('accuracy: {}'.format(acc), file = ofile)

if __name__ == '__main__':
    main()