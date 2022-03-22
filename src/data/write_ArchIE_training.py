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
    keys = list(ifile.keys())

    ofile = open(args.ofile, 'w')

    count = 0
    positive = 0

    for key in keys:
        logging.debug('0: working on key {0} or {1}'.format(keys.index(key) + 1, len(keys)))

        features = np.array(ifile[key]['features'])
        print(features.shape)

        ix = np.sum(features, axis = 1)
        features = features[np.where(ix != 0)[0],:]

        for ix in range(features.shape[0]):
            f = features[ix]
            print(f.shape)
            
            positive += f[-4]
            count += 1

            ofile.write('\t'.join(f.astype(str)) + '\n')

    print(positive)
    print(count)
    print(positive / count)

    ofile.close()

if __name__ == '__main__':
    main()
