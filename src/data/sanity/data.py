# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
import os

pwd = os.path.join(os.path.abspath('.'), 'src/data')

import sys
sys.path.insert(0, pwd)

from data_functions import load_data
from simulate_dros import parameters_df, normalize, writeTbsFile

import glob

import argparse
import logging

import numpy as np
import itertools

def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser()
    # my args
    parser.add_argument("--verbose", action = "store_true", help = "display messages")
    parser.add_argument("--ifile", default = "params.txt")
    
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

SIZE_A = 20
SIZE_B = 14
N_SITES = 10000

def main():
    # let's simulate a sample of reasonable parameters for the Drosophila simulations
    ## -----
    
    args = parse_args()
    
    df = np.loadtxt(args.ifile)
    
    rho = [0.1, 0.25, 0.3]
    migTime = [0.3, 0.2, 0.1]
    migProb = [0.1, 0.2, 0.05]
    
    p = list(itertools.product(rho, migTime, migProb))
    counter = 0
    
    for ix in np.random.choice(range(df.shape[0]), 5, replace = False):
        for p_ in p:
            # simulate the selected parameters
            rho, migTime, migProb = p_
            
            P, ll = parameters_df(df, ix, rho, migTime, migProb, 10)
            
            odir = os.path.join(args.odir, 'iter{0:06d}'.format(counter))
            counter += 1
            
            os.system('mkdir -p {}'.format(odir))
        
            writeTbsFile(P, os.path.join(odir, 'mig.tbs'))
        
            cmd = "cd %s; %s %d %d -t tbs -r tbs %d -I 2 %d %d -n 1 tbs -n 2 tbs -eg 0 1 tbs -eg 0 2 tbs -ma x tbs tbs x -ej tbs 2 1 -en tbs 1 1 -es tbs 2 tbs -ej tbs 3 1 < %s" % (odir, os.path.join(os.getcwd(), 'msmodified/ms'), SIZE_A + SIZE_B, len(P), N_SITES, SIZE_A, SIZE_B, 'mig.tbs')
            p = os.popen(cmd)
            lines = p.readlines()
            
            f = open(os.path.join(odir, 'mig.msOut'), 'w')
            for line in lines:
                f.write(line)
                
            f.close()
            
            ms = os.path.join(odir, 'mig.msOut')
            anc = os.path.join(odir, 'out.anc')
            
            ### read the data back
            x, y, p = load_data(ms, anc)
            
            ### do some sanity checks
            #### -----
            print(counter)
            print(sum([u.shape[1] for u in x]))
            print(sum([u.shape[1] for u in y]))
            
if __name__ == '__main__':
    main()