# -*- coding: utf-8 -*-
import os
import numpy as np
import logging, argparse

from data_functions import *

import h5py

import matplotlib.pyplot as plt
import pickle

def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser()
    # my args
    parser.add_argument("--verbose", action = "store_true", help = "display messages")
    parser.add_argument("--ifile", default = "None")

    parser.add_argument("--downsampling_rate", default = "0.05")
    parser.add_argument("--ofile", default = "None")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
        logging.debug("running in verbose mode")
    else:
        logging.basicConfig(level=logging.INFO)

    return args

def main():
    args = parse_args()

    ifile = h5py.File(args.ifile, 'r')
    ofile = open(args.ofile, 'w')

    count = 0
    positive = 0
    
    dr = float(args.downsampling_rate)

    ii = 0
    while True:
        logging.debug('0: working on key {}...'.format(ii))
        
        try:
            features = np.array(ifile[str(ii)]['features'])
        except:
            break
        
        ii += 1
        features = features[:,0,:,:]
        features = features.reshape(features.shape[0] * features.shape[1], features.shape[-1])

        ix = np.sum(features, axis = 1)
        features = features[np.where(ix != 0)[0],:]
        
        n_samples = int(np.round(len(features)*dr))

        indices = np.random.choice(range(len(features)), n_samples, replace = False)

        for ix in indices:
            f = features[ix, 0]
            
            positive += f[-4]
            count += 1

            ofile.write('\t'.join(f.astype(str)) + '\n')

    print(positive)
    print(count)
    print(positive / count)

    ofile.close()

if __name__ == '__main__':
    main()
