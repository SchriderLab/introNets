# -*- coding: utf-8 -*-
import os
import argparse
import logging

import h5py
import numpy as np

from scipy.special import expit
from evaluate_unet_windowed import cm_analysis
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score, confusion_matrix, accuracy_score

import random
import matplotlib.pyplot as plt
import pandas as pd
import time

def batch_dot(W, x):
    # dot product for a batch of examples
    return np.einsum("ijk,ki->ji", np.tile(W, (len(x), 1, 1)), x.T).T

# use this format to tell the parsers
# where to insert certain parts of the script
# ${imports}

def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser()
    # my args
    parser.add_argument("--verbose", action = "store_true", help = "display messages")
    parser.add_argument("--coef", default = "data/archie_coefs.txt")
    parser.add_argument("--ifile", default = "data/ArchIE_windowed_f_i2.hdf5")

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

def main():
    args = parse_args()

    ifile = h5py.File(args.ifile, 'r')
    keys = list(ifile.keys())
    
    random.shuffle(keys)
    
    coefs = np.loadtxt(args.coef, dtype = object)
    coefs = coefs[:,1]
    coefs[coefs == 'NA'] = '0.0'
    
    coefs = coefs.astype(np.float32)
    
    bias = coefs[0]
    #bias = coefs[0]
    
    w = coefs[1:]
    
    Y = []
    Y_pred = []
    indices = []
    
    logging.info('predicting...')
    for key in keys:
        f = np.array(ifile[key]['f'])[:,:,:-4]
        pos = np.array(ifile[key]['pos']).astype(np.int32)
        y = np.array(ifile[key]['y'])
        
        y_pred = np.zeros(y.shape)
        count = np.zeros(y.shape)
        
        for k in range(f.shape[0]):
            log_prob = batch_dot(w, f[k,:,:]) + bias
            
            i1, i2 = pos[k]
            
            y_pred[:,i1:i2] += log_prob
            count[:,i1:i2] += 1.
            
        y_pred = y_pred[count > 0] / count[count > 0]
        y = y[count > 0]
        
        # save the indices for take-one-out bootstrapping
        i1 = len(Y)
        i2 = i1 + len(y.flatten())
        
        Y.extend(y.flatten())
        Y_pred.extend(expit(y_pred).flatten())
        indices.append((i1, i2))
        
    logging.info('plotting EPS files...')
    # do this for all the examples:
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
    
    logging.info('bootstrapping metrics...')
    
    Y = np.array(Y)
    Y_pred = np.array(Y_pred)
    
    np.savez_compressed(os.path.join(args.odir, 'predictions.npz'), Y = Y, Y_pred = Y_pred)
    
    result = dict()
    result['rocs'] = []
    result['prs'] = []
    result['accs'] = []
    result['t'] = []
    # bootstrap (take-one-out)
    for k in range(len(indices)):
        
        t1 = time.time()
        ix = list(set(range(len(Y))).difference(list(range(indices[k][0], indices[k][1]))))
        
        auroc = roc_auc_score(Y[ix].astype(np.int32), Y_pred[ix])
        aupr = average_precision_score(Y[ix].astype(np.int32), Y_pred[ix])
        acc = accuracy_score(Y[ix].astype(np.int32), np.round(Y_pred[ix]))
        
        result['rocs'].append(auroc)
        result['prs'].append(aupr)
        result['accs'].append(acc)
        result['t'].append(time.time() - t1)
        
        df = pd.DataFrame(result)
        df.to_csv(os.path.join(args.odir, 'estimates.csv'), index = False)
        
    print('auroc: {0} +- {1}'.format(np.mean(result['rocs']), np.std(result['rocs']) / np.sqrt(len(result['rocs'])) * 1.96))
    print('aupr: {0} +- {1}'.format(np.mean(result['prs']), np.std(result['prs']) / np.sqrt(len(result['rocs'])) * 1.96))
    print('accuracy: {0} +- {1}'.format(np.mean(result['accs']), np.std(result['accs']) / np.sqrt(len(result['rocs'])) * 1.96))
        

if __name__ == '__main__':
    main()

