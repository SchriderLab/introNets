# -*- coding: utf-8 -*-

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
    
    idirs = sorted([os.path.join(args.idir, u) for u in os.listdir(args.idir) if ((not '.' in u) and (not 'relate' in u))])
    
    cmd = 'sbatch -n 24 --mem=32G -t 2-00:00:00 --wrap "mpirun python3 src/data/format_relate.py --idir {0} --idir_relate {1} --ofile {2}'
    for idir in idirs:
        cmd_ = cmd.format(idir, idir + '_relate', os.path.join(args.odir, '{0:04d}.hdf5'.format(idirs.index(idir))))
        
        print(cmd_)
        os.system(cmd_)

if __name__ == '__main__':
    main()    