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
    
    parser.add_argument("--topology", default = "knn")
    parser.add_argument("--k", default = "16")
    parser.add_argument("--n_dilations", default = "7")
    parser.add_argument("--sort_pos", action = "store_true")
    
    parser.add_argument("--low_resources", action = "store_true")
    
    parser.add_argument("--pop_sizes", default = "150,156")
    
    parser.add_argument("--ix_y", default = "0")
    parser.add_argument("--low_ram", action = "store_true")

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
    
    idirs = [os.path.join(args.idir, u) for u in os.listdir(args.idir) if ((not '.' in u) and (not 'seedms' in u))]
    
    cmd = 'sbatch -n 24 --mem=32G -t 2-00:00:00 --wrap "mpirun python3 src/data/format_unet_data.py --verbose --topology {4} --idir {0} --odir {1} --ix_y {2}{3} --pop_sizes {5} --k {6} --n_dilations {7}"'
    
    if args.low_ram:
        cmd = 'sbatch -n 24 --mem=8G -t 2-00:00:00 --wrap "mpirun python3 src/data/format_unet_data.py --verbose --topology {4} --idir {0} --odir {1} --ix_y {2}{3} --pop_sizes {5} --k {6} --n_dilations {7}"'
    
    for idir in idirs:
        if args.densify:
            _ = ' --densify'
        else:
            _ = ''
            
        if args.sort_pos:
            _ = _ + ' --sort_pos'
        
        cmd_ = cmd.format(idir, os.path.join(args.odir, idir.split('/')[-1]), args.ix_y, _, args.topology, args.pop_sizes, args.k, args.n_dilations)
        
        os.system(cmd_)

if __name__ == '__main__':
    main()    