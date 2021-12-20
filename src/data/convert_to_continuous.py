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

    x = np.cumsum(np.array(ifile['train']['0']['x_0']), axis = 2) * 4 * np.pi

    x[:,:,0] = 2 * np.pi
    
    x = x.reshape(x.shape[0] * x.shape[1], -1)
    t = np.array(range(x.shape[-1]))
    
    for k in range(len(x)):
        ix = np.where(x[k] != 0)[0]
        t = np.array(range(np.max(ix)))
        
        x[k,:len(t)] = interp1d(t[ix], x[k,ix], kind = 'cubic')(t)
        
    plt.imshow(x)
    plt.savefig('test_output.png', dpi = 100)
    plt.close()
    
    sys.exit()
            
            
    # ${code_blocks}

if __name__ == '__main__':
    main()

