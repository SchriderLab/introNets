# -*- coding: utf-8 -*-
import os
import argparse
import logging

import numpy as np

# use this format to tell the parsers
# where to insert certain parts of the script
# ${imports}

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
    args = parse_args()
    
    idirs = sorted([os.path.join(args.idir, u) for u in os.listdir(args.idir)])

    ret = dict()
    values = []
    
    for idir in idirs:
        params = np.loadtxt(os.path.join(idir, 'mig.tbs'), delimiter = ' ')
        
        P = params[0]
        
        migTime = P[-1]
        T = P[-4]
        
        v = migTime / T
        
        if v in ret.keys():
            ret[v].append(idir)
        else:
            ret[v] = [idir]
        
    values = sorted(list(ret.keys()))
    for ix in range(len(values)):
        odir = os.path.join(args.odir, 'sims{0:03d}'.format(ix))
        
        idirs_ = ret[values[ix]]
        os.system('mkdir -p {}'.format(odir))
        
        for idir in idirs_:
            os.system('cp -r {0} {1}'.format(idir, odir))

if __name__ == '__main__':
    main()

