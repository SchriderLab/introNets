# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
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
import h5py

from sklearn.manifold import Isomap
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

    parser.add_argument("--n_samples", default = "200")

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

    ifiles = [os.path.join(args.idir, u) for u in os.listdir(args.idir) if u.split('.')[-1] == 'hdf5']
    
    segment_lengths = []
    n_samples_per = int(args.n_samples) // len(ifiles)
    
    for ifile in ifiles:
        print(ifile)
        ifile = h5py.File(ifile, 'r')
        keys = list(ifile.keys())
        
        random.shuffle(keys)
        
        for k in range(n_samples_per):
            key = keys[k]
            bp = np.array(ifile[key]['break_points'])
            
            if len(bp) > 0:
                segment_lengths.extend(np.diff(bp))
    
    
    fig = plt.figure(figsize=(8, 8))
    
    ax = fig.add_subplot(111)
    ax.hist(segment_lengths, bins = 35)
    
    print(np.max(segment_lengths))
    print(np.min(segment_length))
    
    plt.savefig(os.path.join(args.odir, 'segl_hist.png'), dpi = 100)
    plt.close()
    
if __name__ == '__main__':
    main()
    