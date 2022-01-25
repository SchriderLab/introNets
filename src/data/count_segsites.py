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

    logging.info('reading .msOut files...')
    segs = os.popen('cat {} | grep segsites'.format(os.path.join(args.idir, '*/*.msOut'))).read()
    
    segs = segs.split('\n')[:-1]
    
    n_sites = [int(u.split(':')[-1]) for u in segs]
    print(np.max(n_sites))
    

if __name__ == '__main__':
    main()
