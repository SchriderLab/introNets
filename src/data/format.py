import os
import argparse
import logging

import h5py
import numpy as np

from data_functions import load_data_dros

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
    
    parser.add_argument("--n_replicates", default = "5000", help = "the number of simulation replicates to include in the output file")
    parser.add_argument("--n_replicates_val", default = "1000", help = "the number of simulation replicates to include for validation")
    
    parser.add_argument("--chunk_size", default = "4")
    
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
        logging.debug("running in verbose mode")
    else:
        logging.basicConfig(level=logging.INFO)

    return args

def main():
    args = parse_args()
    
    idirs = [os.path.join(args.idir, u) for u in os.listdir(args.idir)]
    ofile = h5py.File(args.ofile, 'w')
    
    chunk_size = int(args.chunk_size)
    n_train = int(args.n_replicates)
    n_val = int(args.n_replicates_val)
    
    X1 = []
    X2 = []
    
    ix = 0
    ix_val = 0
    
    for idir in idirs:
        anc_file = os.path.join(idir, 'out.anc')
        ms_file = os.path.join(idir, 'mig.msOut')
        
        if os.path.exists(anc_file) and os.path.exists(ms_file):
            x1, x2, y1, y2, p = load_data_dros(ms_file, anc_file)
            
            X1.extend(x1)
            X2.extend(x2)
            
        while len(X1) > chunk_size * 2:
            if ix * chunk_size < n_train:
                # training
                x1 = np.array(X1[-chunk_size:])
                x2 = np.array(X2[-chunk_size:])
                
                del X1[-chunk_size:]
                del X2[-chunk_size:]
                
                logging.info('writing chunk {} to training set...'.format(ix))
                ofile.create_dataset('train/{}/x1'.format(ix), data = np.array(x1, dtype = np.uint8), compression = 'lzf')
                ofile.create_dataset('train/{}/x2'.format(ix), data = np.array(x2, dtype = np.uint8), compression = 'lzf')
                
                ix += 1
            
            if ix_val * chunk_size < n_val:
                # validation
                x1 = np.array(X1[-chunk_size:])
                x2 = np.array(X2[-chunk_size:])
                
                del X1[-chunk_size:]
                del X2[-chunk_size:]
                
                logging.info('writing chunk {} to validation set...'.format(ix_val))
                ofile.create_dataset('train/{}/x1'.format(ix_val), data = np.array(x1, dtype = np.uint8), compression = 'lzf')
                ofile.create_dataset('train/{}/x2'.format(ix_val), data = np.array(x2, dtype = np.uint8), compression = 'lzf')
                
                ix_val += 1
                
        if (ix * chunk_size >= n_train) and (ix_val * chunk_size >= n_val):
            break
        
    ofile.close()
        
    # ${code_blocks}

if __name__ == '__main__':
    main()
# -*- coding: utf-8 -*-

