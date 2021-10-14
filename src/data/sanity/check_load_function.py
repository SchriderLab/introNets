# -*- coding: utf-8 -*-
import os

pwd = os.path.join(os.path.abspath('.'), 'src/data')

import sys
sys.path.insert(0, pwd)

from data_functions import load_data
import glob

import argparse
import logging

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
    
    msFiles = sorted(glob.glob(os.path.join(args.idir, '*.txt')))
    ancFiles = sorted(glob.glob(os.path.join(args.idir, '*.anc')))
    
    for ix in range(len(msFiles)):
        x, y = load_data(msFiles[ix], ancFiles[ix])
        
        print(x.shape, y.shape)
    
    # ${code_blocks}

if __name__ == '__main__':
    main()
