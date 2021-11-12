# -*- coding: utf-8 -*-
import os
import argparse
import logging

import itertools

# use this format to tell the parsers
# where to insert certain parts of the script
# ${imports}

def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser()
    # my args
    parser.add_argument("--verbose", action = "store_true", help = "display messages")
    # ${args}

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
    
    dense = [False]
    k = [4, 8, 12]
    n_dilations = [5, 7, 11]
    topologies = ['knn']
    
    todo = list(itertools.product(dense, k, n_dilations, topologies))
    cmd = 'sbatch -t 1-00:00:00 --mem=32G -n 24 --wrap "mpirun python3 src/data/change_topology.py --idir {0} --odir {1} --k {2} --n_dilations {3}{4}"'
    
    for ix in range(len(todo)):
        densify, k_, n_dilations_, topology = todo[ix]
        
        if densify:
            _ = ' --densify'
        else:
            _ = ''
            
        odir = os.path.join(args.odir, 'k{0}_d{1}_{2}'.format(k_, n_dilations_, topology))
        
        cmd_ = cmd.format(args.idir, odir, k_, n_dilations_, _)
        print(cmd_)
        
        os.system(cmd_)
        

    # ${code_blocks}

if __name__ == '__main__':
    main()

