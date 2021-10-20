# -*- coding: utf-8 -*-

import logging
import argparse
import os

def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser()
    # my args
    parser.add_argument("--verbose", action = "store_true", help = "display messages")
    parser.add_argument("--idir", default = "None")
    
    parser.add_argument("--odir", default = "None")
    parser.add_argument("--densify", action = "store_true")
    
    parser.add_argument("--ix_y", default = "0")

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
        logging.debug("running in verbose mode")
    else:
        logging.basicConfig(level=logging.INFO)

    if args.odir != "None":
        if not os.path.exists(args.odir):
            os.mkdir(args.odir)
            logging.debug('root: made output directory {0}'.format(args.odir))
        else:
            os.system('rm -rf {0}'.format(os.path.join(args.odir, '*')))

    return args

def main():
    args = parse_args()
    
    idirs = [os.path.join(args.idir, u) for u in os.listdir(args.idir) if not '.' in u]
    
    cmd = 'sbatch -n 24 --mem=32G -t 2-00:00:00 --wrap "mpirun python3 src/data/format_unet_data.py --verbose --idir {0} --odir {1} --ix_y {2}"'
    for idir in idirs:
        cmd_ = cmd.format(idir, os.path.join(args.odir, idir.split('/')[-1]), args.ix_y)
        if args.densify:
            cmd_ = cmd_ + " --densify"
        
        os.system(cmd_)

if __name__ == '__main__':
    main()    