# -*- coding: utf-8 -*-
import os
import argparse
import logging

import glob
import numpy as np

from data_functions import load_data

from itertools import groupby
from operator import itemgetter

import matplotlib
matplotlib.use('Agg')

import pandas as pd

import matplotlib.pyplot as plt
from mpi4py import MPI
# use this format to tell the parsers
# where to insert certain parts of the script
# ${imports}

def find_blocks(X):
    X = np.mean(X, axis = 0)
    one_blocks = []
    
    # 1 blocks ## (unrealistic introgression)
    data = list(np.where(X == 1)[0])

    for k, g in groupby(enumerate(data), lambda ix : ix[0] - ix[1]):
         _ = list(map(itemgetter(1), g))
         
         if len(_) >= 2:
             one_blocks.append(len(_))
             
    # proportion of:
    p1 = sum(one_blocks) / len(X)
             
    # more realistic introgression
    p2 = len(np.where(X > 0)[0]) / len(X)
    
    return p1, p2
        

def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser()
    # my args
    parser.add_argument("--verbose", action = "store_true", help = "display messages")
    parser.add_argument("--idir", default = "None")

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
    # configure MPI
    comm = MPI.COMM_WORLD
    
    args = parse_args()

    msFiles = sorted(glob.glob(os.path.join(args.idir, '*/mig.msOut')))[:32]
    ancFiles = sorted(glob.glob(os.path.join(args.idir, '*/out.anc')))[:32]
    
    result = dict()
    result['migTime'] = []
    result['migProb'] = []
    result['rho'] = []
    result['theta'] = []
    result['nu_a'] = []
    result['nu_b'] = []
    result['alpha1'] = []
    result['alpha2'] = []
    result['T'] = []
    result['ratio'] = []
    
    for ix in range(comm.rank, len(msFiles), comm.size):        
        msFile = msFiles[ix]
        ancFile = ancFiles[ix]
        
        logging.info('{0}: working on {1}...'.format(comm.rank, msFile))
        
        x, y, p = load_data(msFile, ancFile)
        
        theta = p[0]
        rho = p[1]
        nu_a = p[2]
        nu_b = p[3]
        alpha1 = p[4]
        alpha2 = p[5]
        T = p[8]
        migTime = p[10]
        migProb = 1 - p[11]
        
        _ = []
        for k in range(len(y)):
            y_ = y[k]
            
            if len(y_.shape) == 2:
                p1, p2 = find_blocks(y_[20:,:])
            else:
                continue
            
            # filter no introgression case
            if p1 + p2 != 0 and p1 != 0:
                _.append(p2 / p1)
                
        result['migTime'].append(migTime)
        result['migProb'].append(migProb)
        result['rho'].append(rho)
        result['theta'].append(theta)
        result['nu_a'].append(nu_a)
        result['nu_b'].append(nu_b)
        result['alpha1'].append(alpha1)
        result['alpha2'].append(alpha2)
        result['T'].append(T)
        result['ratio'].append(np.mean(_))
        
    comm.Barrier()
    
    if comm.rank == 0:
        result = comm.gather(result, root = 0)
        
        _ = dict()
        for v in result:
            for key in list(v.keys()):
                if key not in _.keys():
                    _[key] = v[key]
                else:
                    _[key].extend(v[key])
                    
        df = pd.DataFrame(_)
        df.to_csv('blocks.csv', index = False)
        
        plt.hist(_['ratio'], bins = 35)
        plt.savefig('block_hist.png', dpi = 100)
        plt.close()
            
        

if __name__ == '__main__':
    main()

