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

    cmd = 'python3 src/data/simulate_msmodified.py --mt_range {0} --t_range {1} --odir {2} --n_jobs 25 --n_samples 100 --slurm --ifile params.txt'
    
    st = np.linspace(45000, 105000, 5)
    mt = np.array([0.  , 0.05, 0.1 , 0.15, 0.2 ])
    
    for i in range(len(st) - 1):
        for j in range(len(mt) - 1):
            odir = os.path.join(args.odir, '{0}_{1}'.format(i, j))
            
            cmd_ = cmd.format('{0},{1}'.format(mt[j], mt[j + 1]), '{0},{1}'.format(st[i], st[i + 1]), odir)
            
            print(cmd_)
            os.system(cmd_)
            
    # ${code_blocks}

if __name__ == '__main__':
    main()

