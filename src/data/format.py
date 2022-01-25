import os
import argparse
import logging

import h5py
import numpy as np

from data_functions import load_data_dros, load_npz
import random

# use this format to tell the parsers
# where to insert certain parts of the script
# ${imports}

def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser()
    # my args
    parser.add_argument("--verbose", action = "store_true", help = "display messages")
    parser.add_argument("--idir", default = "None", help = "directory with simulation directories in it")
    
    parser.add_argument("--ofile", default = "None", help = "hdf5 file for pre-chunked data to go into")
    
    parser.add_argument("--n_replicates", default = "100000", help = "the number of simulation replicates to include in the output file")
    parser.add_argument("--n_replicates_val", default = "10000", help = "the number of simulation replicates to include for validation")
    
    parser.add_argument("--chunk_size", default = "4")
    parser.add_argument("--n_sites", default = "508")
    
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
        logging.debug("running in verbose mode")
    else:
        logging.basicConfig(level=logging.INFO)

    return args

def main():
    args = parse_args()
    
    idirs = sorted([os.path.join(args.idir, u) for u in os.listdir(args.idir)])
    random.shuffle(idirs)
    
    ofile = h5py.File(args.ofile, 'w')
    
    chunk_size = int(args.chunk_size)
    n_train = int(args.n_replicates)
    n_val = int(args.n_replicates_val)
    
    X = []
    P = []
    
    ix = 0
    ix_val = 0
    
    for idir in idirs:
        anc_file = os.path.join(idir, 'out.anc')
        ms_file = os.path.join(idir, 'mig.msOut')
        
        if os.path.exists(ms_file):
            try:
                x, p = load_data_dros(ms_file, anc_file, n_sites = int(args.n_sites))
            except:
                continue
            
            X.extend(x)
            P.extend(p)
            
        while len(X) > chunk_size * 2:
            if ix * chunk_size < n_train:
                # training
                x = np.array(X[-chunk_size:])
                p = np.array(P[-chunk_size:])
                
                del X[-chunk_size:]
                del P[-chunk_size:]
                
                logging.info('writing chunk {} to training set...'.format(ix))
                ofile.create_dataset('train/{}/x_0'.format(ix), data = np.array(x, dtype = np.uint8), compression = 'lzf')
                ofile.create_dataset('train/{}/p'.format(ix), data = p, compression = 'lzf')
                
                ix += 1
            
            if ix_val * chunk_size < n_val:
                # training
                x = np.array(X[-chunk_size:])
                p = np.array(P[-chunk_size:])
                
                del X[-chunk_size:]
                del P[-chunk_size:]
                
                logging.info('writing chunk {} to training set...'.format(ix))
                ofile.create_dataset('train/{}/x_0'.format(ix), data = np.array(x, dtype = np.uint8), compression = 'lzf')
                ofile.create_dataset('train/{}/p'.format(ix), data = p, compression = 'lzf')
                
                ix_val += 1
                
            if (ix * chunk_size >= n_train) and (ix_val * chunk_size >= n_val):
                break
                
        if (ix * chunk_size >= n_train) and (ix_val * chunk_size >= n_val):
            break
        
    ofile.close()
        
    # ${code_blocks}

if __name__ == '__main__':
    main()
# -*- coding: utf-8 -*-

