# -*- coding: utf-8 -*-
import os
import argparse
import logging
import h5py

# use this format to tell the parsers
# where to insert certain parts of the script
# ${imports}

from scipy.interpolate import interp1d
import numpy as np
import matplotlib
matplotlib.use('Agg')

import sys

import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, cdist
from seriate import seriate

from scipy.optimize import linear_sum_assignment

def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser()
    # my args
    parser.add_argument("--verbose", action = "store_true", help = "display messages")
    parser.add_argument("--ifile", default = "None")
    parser.add_argument("--ofile", default = "None")

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
    #ofile = h5py.File(args.ofile, 'w')
    
    #train_keys = list(ifile['train'].keys())
    #val_keys = list(ifile['val'].keys())

    x = np.array(ifile['train']['0']['x_0'])[0,:,:]
    y = np.array(ifile['train']['0']['y'])[0,:,:]
    x[:,-1] = 1
    
    x = np.cumsum(x, axis = 1) * 2 * np.pi

    
    mask = np.zeros(x.shape)
    ix = list(np.where(np.diff(x) != 0))
    ix[-1] += 1
    mask[tuple(ix)] = 1
    mask[:,-1] = 1
    
    x[mask == 0] = 0
    t = np.array(range(x.shape[-1]))
    
    for k in range(len(x)):
        ix = [0] + list(np.where(x[k] != 0)[0])
        print(len(ix))
        
        t = np.array(range(np.max(ix)))
        
        if len(ix) > 3:
            x[k,:len(t)] = interp1d(ix, x[k,ix], kind = 'cubic')(t)
            
            
    x = np.cos(x)
    x1 = x[:150,:]
    x2 = x[150:300,:]
    
    D = pdist(x1, metric = 'euclidean')
    ix = seriate(D)

    x1 = x1[ix, :]

    x_im = np.ones((150, 128, 3))
    D = cdist(x1, x2, metric = 'euclidean')
    i, j = linear_sum_assignment(D)
    
    x2 = x2[j,:]
    
    x_im[:,:,0] = x1
    x_im[:,:,1] = x2
    
    plt.imshow(x)
    plt.savefig('test_output.png', dpi = 100)
    plt.close()
    
    sys.exit()
            
            
    # ${code_blocks}

if __name__ == '__main__':
    main()

