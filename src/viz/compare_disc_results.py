# -*- coding: utf-8 -*-
import os
import argparse
import logging

import numpy as np
import pickle

from scipy.special import softmax

from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# use this format to tell the parsers
# where to insert certain parts of the script
# ${imports}

def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser()
    # my args
    parser.add_argument("--verbose", action = "store_true", help = "display messages")
    parser.add_argument("--ifile_a", default = "None")  # new numpy file
    parser.add_argument("--ifile_b", default = "None") # old pkl file

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

    ifile_a = np.load(args.ifile_a)
    ifile_b = pickle.load(open(args.ifile_b, 'rb'))
    
    Ya = softmax(ifile_a['Y'], axis = 1)
    Yb = np.array(ifile_b[0])
    
    Pa = ifile_a['P']
    Pb = []
    
    for r in ifile_b[1]:
        if len(r) > 1:
            Pb.append([r[0], r[-1]])
        else:
            Pb.append([-1, -1])
            
    Pb = np.array(Pb)
    
    Xa = np.mean(Pa, axis = 1)
    ya = Ya[:,1]
    
    Xb = []
    yb = []
    for k in range(Pb.shape[0]):
        if Pb[k][0] != -1:
            Xb.append(np.mean(Pb[k]))
            
            yb.append(Yb[k][1])
            
    fa = interp1d(Xa, ya)
    fb = interp1d(Xb, yb)
    
    x = np.linspace(max([np.min(Xa), np.min(Xb)]), min([np.max(Xa), np.max(Xb)]), 100)
    
    ya = fa(x)
    yb = fb(x)
    
    plt.scatter(ya, yb)
    plt.show()
    
if __name__ == '__main__':
    main()

