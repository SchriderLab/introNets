# -*- coding: utf-8 -*-
import os
import argparse
import logging

import random
import glob
import numpy as np

from scipy.spatial.distance import pdist, squareform

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

from scipy.sparse.linalg import eigs
from sklearn.metrics import pairwise_distances
# use this format to tell the parsers
# where to insert certain parts of the script
# ${imports}

def seriate_spectral(x):    
    C = pairwise_distances(x)
    C[np.where(np.isnan(C))] = 0.

    C = np.diag(C.sum(axis = 1)) - C
    _, v = eigs(C, k = 2, which = 'SM')

    f = v[:,1]
    ix = np.argsort(np.real(f))

    x = x[ix,:]
    
    return x, ix

def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser()
    # my args
    parser.add_argument("--verbose", action = "store_true", help = "display messages")
    parser.add_argument("--idir", default = "None")

    parser.add_argument("--n_sites", default = "128")
    parser.add_argument("--n_samples", default = "10000")

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

    models = os.listdir(args.idir)
    n_sites = int(args.n_sites)
    n_samples = int(args.n_samples)
    
    result = dict()
    result['model'] = []
    result['entropy'] = []
    result['entropy_A'] = []
    result['entropy_B'] = []
    
    for model in models:
        ix = models.index(model)
        
        logging.info('working on {}...'.format(model))
        
        X1 = []
        X2 = []
        Y = []
        
        path_lengths_1 = []
        path_lengths_2 = []
        path_lengths = []
        
        idir = os.path.join(args.idir, model)
        ifiles = glob.glob(os.path.join(idir, '*/*.npz'))
        random.shuffle(ifiles)
        
        while len(X1) + len(X2) < n_samples:
            ifile = ifiles[0]
            ifile = np.load(ifile)
            
            del ifiles[0]
            
            x = ifile['x'][:,:n_sites]
            y = ifile['y'][:,:n_sites]
            
            # how well does spectral seriation work?
            x1, _ = seriate_spectral(x[:150,:])
            dx1 = np.sum(np.linalg.norm(np.diff(x1, axis = 0), axis = 0))
            
            x2, _ = seriate_spectral(x[150:,:])
            dx2 = np.sum(np.linalg.norm(np.diff(x2, axis = 0), axis = 0))
            
            path_lengths_1.append(dx1)
            path_lengths_2.append(dx2)
            
            x1, _ = seriate_spectral(x)
            dx1 = np.sum(np.linalg.norm(np.diff(x1, axis = 0), axis = 0))
            
            path_lengths.append(dx1)
            
            X1.extend(list(x[:150]))
            X2.extend(list(x[150:]))
            Y.extend(list(y))
            
        X1 = np.array(X1)
        X2 = np.array(X2)
        
        Y = np.array(Y)
        
        # we'll make a distance figure
        fig, axes = plt.subplots(nrows = 3, ncols = 2, sharex = True)
        
        ## ====================
        #### Euclidean distance histograms (inner and outer)
        # distances (pop A)
        Xd = pdist(X1, metric = 'euclidean', p = 2)
        entropy_A = np.mean(np.log(X1.shape[0] * np.min(squareform(Xd) + 10e-5, axis = 0)))
        
        # plot a sample as a histogram (let's say the original sample size)
        Xd_ = np.random.choice(Xd, n_samples, replace = False)
        axes[0, 0].hist(Xd_, bins = 35, color = 'k')
        axes[0, 0].set_title('pdist (pop A)')
        
        # distances (pop A)
        Xd = pdist(X2, metric = 'euclidean', p = 2)
        entropy_B = np.mean(np.log(X1.shape[0] * np.min(squareform(Xd) + 10e-5, axis = 0)))
        
        # plot a sample as a histogram (let's say the original sample size)
        Xd_ = np.random.choice(Xd, n_samples, replace = False)
        axes[1, 0].hist(Xd_, bins = 35, color = 'k')
        axes[1, 0].set_title('pdist (pop B)')
        
        # distances (pop A)
        Xd = pdist(np.concat([X1, X2]), metric = 'euclidean', p = 2)
        entropy = np.mean(np.log((X1.shape[0] + X2.shape[0]) * np.min(squareform(Xd) + 10e-5, axis = 0)))
        
        # plot a sample as a histogram (let's say the original sample size)
        Xd_ = np.random.choice(Xd, n_samples, replace = False)
        axes[2, 0].hist(Xd_, bins = 35, color = 'k')
        axes[2, 0].set_title('pdist')
        
        
        ## ====================
        #### Path length (for spectral seriation)
        
        axes[0, 1].hist(path_lengths_1, bins = 35)
        axes[0, 1].set_title('spectral pl (pop A)')
        
        axes[1, 1].hist(path_lengths_2, bins = 35)
        axes[1, 1].set_title('spectral pl (pop B)')
        
        axes[2, 1].hist(path_lengths, bins = 35)
        axes[2, 1].set_title('spectral pl')
        
        plt.savefig(os.path.join(args.odir, '{0}_euclidean.png'.format(model)), dpi = 200)
        plt.close()
        
        

        
        
        
        
            
            
            
        
        
    # ${code_blocks}

if __name__ == '__main__':
    main()

