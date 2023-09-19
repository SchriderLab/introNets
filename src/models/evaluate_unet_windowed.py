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
from scipy.interpolate import interp1d

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
    parser.add_argument("--ifile", default = "None", help = "h5 file with formatted genomes in it to predict on. produced with src/data/format_genomes.py")
    parser.add_argument("--weights", default = "None", help = "weights of the pretrained model") 
    parser.add_argument("--n_classes", default = "1", help = "how many populations (image channels) to return in the y-prediction. must match the number trained upon")
    parser.add_argument("--sigma", default = "30", help = "std of the gaussian window if smoothing the predictions (use --smooth)")
    parser.add_argument("--n", default = "128", help = "size of the sliding window.  must be the same as passed to format_genomes and the training routine")
    
    parser.add_argument("--smooth", action = "store_true", help = "whether to use a guassian kernel to smooth the predictions. puts less weight on the pixels toward the edges of the prediction in the position axis")

    parser.add_argument("--pop", default = "None", help = "use to specify the pop that's being predicted")
    parser.add_argument("--plot", action = "store_true")
    parser.add_argument("--n_to_plot", default = "None")
    
    parser.add_argument("--save_pred", action = "store_true")

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
        else:
            os.system('rm -rf {}'.format(os.path.join(args.odir, '*')))

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
    
    Y = []
    Y_pred = []
    indices_ = []
    
    counter = 0
    
    accuracies = []
    auprs = []
    rocs = []
    
    M = np.zeros((2, 2))
    
    if args.save_pred:
        os.system('mkdir -p {}'.format(os.path.join(args.odir, 'preds')))
    
    if args.pop != "None":
        pop = int(args.pop)
    else:
        pop = None
        
    pr_thresholds = np.linspace(0., 1., 100)
    roc_thresholds = np.linspace(0., 2., 100)
    
    fpr_ = np.zeros((100,))
    tpr_ = np.zeros((100,))
    
    recall_ = np.zeros((100,))
    precision_ = np.zeros((100,))
    
    logging.info('predicting on {} keys...'.format(len(keys)))
    for key in keys:
        indices = np.array(ifile[key]['indices'])
        ix = np.array(ifile[key]['ix'])
        x = np.array(ifile[key]['x_0']).squeeze()
        y = np.squeeze(np.array(ifile[key]['y']))
        pos = np.array(ifile[key]['pos'])
         
        
        l = np.max(ix) + 1
        n = indices.shape[-1]
        
        # single channel case
        if len(y.shape) == 3:
            y = np.expand_dims(y, 1)
        
        # get an array to store the results and the count
        y_pred = np.zeros((y.shape[-3], y.shape[-2], l), dtype = np.float32)
        y_true = np.zeros((y.shape[-3], y.shape[-2], l), dtype = np.float32)
        count = np.zeros((y.shape[-3], y.shape[-2], l), dtype = np.float32)
        
        ii = list(range(x.shape[0]))
        
        y_pred_ = []
        for c in chunks(ii, bs):
            x_ = torch.FloatTensor(x[c]).to(device)
            
            with torch.no_grad():
                y_ = model(x_)
                
                if args.smooth:
                    y_ = y_ * G
                
            y_ = y_.detach().cpu().numpy()
            if x_.shape[0] == 1:
                y_ = np.expand_dims(y_, 0)
            
            
            y_pred_.append(y_)
            
        y_pred_ = np.concatenate(y_pred_)
        
        if len(y_pred_.shape) == 3:
            y_pred_ = np.expand_dims(y_pred_, 1)
        
        if len(indices.shape) == 4:
            indices = indices[:,0,:,:]
        
        if pop is not None:
            indices = indices[:,[pop],:]
        
        for j in range(y.shape[-3]):
            for k in range(y_pred_.shape[0]):
                # the seriated order
                ii = indices[k,j,:]
                # list of the index of unique individuals if there was upsampling
                ii_u = uni(ii)

                # go down to only those unique individuals
                y_ = y_pred_[k][j][ii_u]
                
                # used to sort back
                ii = [ii[u] for u in ii_u]
                ii = np.argsort(ii)
                
                ix_ = ix[k]
                y_pred[j][:,ix_] += y_[ii]
                
                if args.smooth:
                    count[j][:,ix_] += Gn.flatten()
                else:
                    count[j][:,ix_] += 1.
                
                y_ = y[k,j,ii_u]
                y_ = y_[ii]
                
                y_true[j][:,ix_] = y_
        

        ii_ = np.where(count != 0)[-1]

        y_pred = y_pred[:,:,ii_]
        count = count[:,:,ii_]
        y_true = y_true[:,:,ii_]
        
        y_pred = expit(y_pred / count)        
        
        if args.save_pred:
            np.savez_compressed(os.path.join(os.path.join(args.odir, 'preds'), '{0:03d}.npz'.format(int(key))), y_pred = y_pred, y = y_true, pos = pos)
        
        """
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
        
        accuracies.append(np.mean(np.abs(y_true - y_pred_round)))
        auprs.append(average_precision_score(y_true, y_pred))
        
        try:
            rocs.append(roc_auc_score(y_true, y_pred))
            
            fpr, tpr, thresholds_roc = roc_curve(list(map(int, y_true)), y_pred)
            precision, recall, thresholds_pr = precision_recall_curve(list(map(int, y_true)), y_pred)
        
            fpr = [0.] + list(fpr) + [1.]
            tpr = [0.] + list(tpr) + [1.]
            thresholds_roc = [0.] + list(thresholds_roc) + [2.]
            
            recall = [0.] + list(recall)
            precision = [1.] + list(precision)
            thresholds_pr = [0.] + list(thresholds_pr) + [1.]

    
            fpr_ += interp1d(thresholds_roc, fpr)(roc_thresholds)
            tpr_ += interp1d(thresholds_roc, tpr)(roc_thresholds)
            
            recall_ += interp1d(thresholds_pr, recall)(pr_thresholds)
            precision_ += interp1d(thresholds_pr, precision)(pr_thresholds)
            
            counter += 1
        except:
            rocs.append(np.nan)
        """
        if (counter + 1) % 10 == 0:
            print(counter)
        
        
    print(np.nanmean(accuracies), np.nanstd(accuracies) * 1.96 / np.sqrt(1000))
    print(np.nanmean(auprs), np.nanstd(auprs) * 1.96 / np.sqrt(1000))  
    print(np.nanmean(rocs), np.nanstd(rocs) * 1.96 / np.sqrt(1000)) 
    
    logging.info('plotting EPS files...')
    # do this for all the examples:
    cm_m(M, os.path.join(args.odir, 'confusion_matrix.eps'), ['not introgressed', 'introgressed'])
    
    recall = recall_ / counter
    precision = precision_ / counter
    
    fpr = fpr_ / counter
    tpr = tpr_ / counter
    
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
