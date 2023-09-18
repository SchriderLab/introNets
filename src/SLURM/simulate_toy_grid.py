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
    parser.add_argument("--odir", default = "None")
    
    parser.add_argument("--direction", default = "ab")
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

    cmd = 'python3 src/data/simulate_slim.py --st {0} --mt {1} --odir {2} --n_jobs 250 --slurm --direction {3}'
    
    st = np.linspace(2., 6., 5)
    mt = np.linspace(0.125, 0.375, 5)
    
    for i in range(len(st)):
        for j in range(len(mt)):
            odir = os.path.join(args.odir, '{0}_{1}'.format(i, j))
            
            cmd_ = cmd.format(st[i], mt[j], odir, args.direction)
            os.system(cmd_)
            
    # ${code_blocks}

if __name__ == '__main__':
    main()
