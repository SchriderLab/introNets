# -*- coding: utf-8 -*-
import os
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
    parser.add_argument("--out_shape", default = "2,64,128")
    parser.add_argument("--pop_sizes", default = "64,64")
    parser.add_argument("--pop", default = "1")
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

    idirs = os.listdir(args.idir)
    cmd = 'sbatch -n 32 --mem=32G --partition=dschridelab --constraint=rhel8 -t 2-00:00:00 --wrap "mpirun python3 src/data/format.py --idir {0} --out_shape {1} --pop_sizes {2} --ofile {3} --pop {4}"'

    for idir in idirs:
        idir_ = os.path.join(args.idir, idir)
        ofile = os.path.join(args.idir, '{}.hdf5'.format(idir))
        
        cmd_ = cmd.format(idir_, args.out_shape, args.pop_sizes, ofile, args.pop)
        
        print(cmd_)
        os.system(cmd_)

if __name__ == '__main__':
    main()
