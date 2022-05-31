# -*- coding: utf-8 -*-
import os
import argparse
import logging

import h5py
import pickle
import numpy as np
import random

# use this format to tell the parsers
# where to insert certain parts of the script
# ${imports}

def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser()
    # my args
    parser.add_argument("--verbose", action = "store_true", help = "display messages")
    parser.add_argument("--ifile", default = "None")
    parser.add_argument("--n_samples", default = "5000")
    
    # ${args}

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
    n_keys = len(keys)
    
    random.shuffle(keys)
    
    y_means = []
    
    for key in keys[:int(args.n_samples)]:
        y = np.array(ifile[key]['y'])
        chunk_size = y.shape[0]
        
        y_means.append(np.mean(y))

    ratio = (1 - np.mean(y_means)) / np.mean(y_means)
    print('info for {}'.format(args.ifile))
    print('neg / pos ratio: {}'.format(ratio))
    print('chunk_size: {}'.format(chunk_size))
    print('n_replicates: {}'.format(n_keys * chunk_size))
    # ${code_blocks}

if __name__ == '__main__':
    main()

# -*- coding: utf-8 -*-

