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
import random

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

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]
        
def cm_m(m, filename, labels, figsize=(10,10)):
    cm = m
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
    cm = pd.DataFrame(cm, index=labels, columns=labels)
    cm.index.name = 'Actual'
    cm.columns.name = 'Predicted'
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm, annot=annot, fmt='', ax=ax)
    plt.savefig(filename)
    plt.close()

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
    cm = pd.DataFrame(cm, index=labels, columns=labels)
    cm.index.name = 'Actual'
    cm.columns.name = 'Predicted'
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm, annot=annot, fmt='', ax=ax)
    plt.savefig(filename)
    plt.close()


# use this format to tell the parsers
# where to insert certain parts of the script
# ${imports}

def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser()
    # my args
    parser.add_argument("--verbose", action = "store_true", help = "display messages")
    parser.add_argument("--ifile", default = "None")
    parser.add_argument("--weights", default = "None") # weights of the pre-trained model
    parser.add_argument("--n_classes", default = "1")
    parser.add_argument("--sigma", default = "30")
    parser.add_argument("--n", default = "192")

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

def uni(seq):
    seen = set()
    seen_add = seen.add
    return [k for k in range(len(seq)) if not (seq[k] in seen or seen_add(seq[k]))]

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
    
    bs = 32
    ifile = h5py.File(args.ifile, 'r')
    
    keys = list(ifile.keys())
    random.shuffle(keys)
    
    G = gaussian(int(args.n), int(args.sigma))
    
    G = G.view(1, 1, int(args.n)).to(device)
    Gn = G.detach().cpu().numpy()
    
    plt.plot(Gn.flatten())
    plt.show()
    
    Y = []
    Y_pred = []
    indices_ = []
    
    counter = 0
    
    accuracies = []
    auprs = []
    rocs = []
    
    M = np.zeros((2, 2))
    
    fprs = []
    tprs = []
    
    logging.info('predicting on {} keys...'.format(len(keys)))
    for key in keys:
        indices = np.array(ifile[key]['indices'])
        ix = np.array(ifile[key]['ix'])
        x = np.array(ifile[key]['x_0'])
        y = np.squeeze(np.array(ifile[key]['y']))
        
        l = np.max(ix) + 1
        n = indices.shape[-1]
        
        # get an array to store the results and the count
        y_pred = np.zeros((100, l), dtype = np.float32)
        y_true = np.zeros((100, l), dtype = np.float32)
        count = np.zeros((100, l), dtype = np.float32)
    
        ii = list(range(x.shape[0]))
        
        y_pred_ = []
        for c in chunks(ii, bs):
            x_ = torch.FloatTensor(x[c]).to(device)
            
            print(x_.shape)
            with torch.no_grad():
                y_ = model(x_)
                
            y_ = y_.detach().cpu().numpy()
            if len(y_.shape) == 2:
                y_ = np.expand_dims(y_, 0)
            
            y_pred_.append(y_)
            
        y_pred_ = np.concatenate(y_pred_)
        for k in range(y_pred_.shape[0]):
            ii = indices[k,0,:]
            ii_u = uni(ii)

            y_ = y_pred_[k][ii_u]
            ii = [ii[u] for u in ii_u]
            ii = np.argsort(ii)
            
            ix_ = ix[k]
            
            y_pred[:,ix_] += y_[ii]
            count[:,ix_] += 1.
            
            y_ = y[k,ii_u]
            y_ = y_[ii]
            
            y_true[:,ix_] = y_
        
        y_pred = expit(y_pred / count)        
        counter += 1
        
        if (counter + 1) % 10 == 0:
            print(counter)
        
        # save the indices for take-one-out bootstrapping
        i1 = len(Y)
        i2 = i1 + len(y.flatten())
        
        y_pred = y_pred.flatten()
        y_pred_round = np.round(y_pred.flatten())
        y_true = y_true.flatten()
        
        ii = np.where((y_pred_round == y_true) & (y_true == 0))[0]
        M[0,0] += len(ii)
        
        ii = np.where((y_pred_round == y_true) & (y_true == 1))[0]
        M[1,1] += len(ii)
        
        ii = np.where((y_pred_round != y_true) & (y_true == 0))[0]
        M[0,1] += len(ii)
        
        ii = np.where((y_pred_round != y_true) & (y_true == 1))[0]
        M[1,0] += len(ii)
        
        accuracies.append(accuracy_score(y_true, y_pred_round))
        auprs.append(average_precision_score(y_true, y_pred))
        
        try:
            rocs.append(roc_auc_score(y_true, y_pred))
            """
            fpr, tpr, _ = roc_curve(list(map(int, y_true)), y_pred)
        
            fprs.append(fpr)
            tprs.append(tpr)
        
            print(len(fpr), len(tpr))
            """
        except:
            rocs.append(np.nan)
        
        
        
    print(np.nanmean(accuracies), np.nanstd(accuracies) * 1.96 / np.sqrt(1000))
    print(np.nanmean(auprs), np.nanstd(auprs) * 1.96 / np.sqrt(1000))  
    print(np.nanmean(rocs), np.nanstd(rocs) * 1.96 / np.sqrt(1000)) 
    
    logging.info('plotting EPS files...')
    # do this for all the examples:
    cm_m(M, os.path.join(args.odir, 'confusion_matrix.eps'), ['not introgressed', 'introgressed'])
    sys.exit()
    
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
    
    """
    logging.info('bootstrapping metrics...')
    
    Y = np.array(Y)
    Y_pred = np.array(Y_pred)
    
    rocs = []
    prs = []
    accs = []
    # bootstrap (take-one-out)
    for k in range(len(indices)):
        ix = list(set(range(len(Y))).difference(list(range(indices_[k][0], indices_[k][1]))))
        
        auroc = roc_auc_score(Y[ix].astype(np.int32), Y_pred[ix])
        aupr = average_precision_score(Y[ix].astype(np.int32), Y_pred[ix])
        acc = accuracy_score(Y[ix].astype(np.int32), np.round(Y_pred[ix]))
        
        rocs.append(auroc)
        prs.append(aupr)
        accs.append(acc)
        
    print('auroc: {0} +- {1}'.format(np.mean(rocs), np.std(rocs) / np.sqrt(len(rocs)) * 1.96))
    print('aupr: {0} +- {1}'.format(np.mean(prs), np.std(prs) / np.sqrt(len(rocs)) * 1.96))
    print('accuracy: {0} +- {1}'.format(np.mean(accs), np.std(accs) / np.sqrt(len(rocs)) * 1.96))
    """

if __name__ == '__main__':
    main()
