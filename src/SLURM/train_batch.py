# -*- coding: utf-8 -*-
import os
import argparse
import logging

import glob

# use this format to tell the parsers
# where to insert certain parts of the script
# ${imports}

def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser()
    # my args
    parser.add_argument("--verbose", action = "store_true", help = "display messages")
    parser.add_argument("--idir", default = "None")
    parser.add_argument("--config", default = "None")

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

    ifiles = glob.glob(os.path.join(args.idir, '*.hdf5'))
    cmd = 'sbatch -n 4 --mem=16G --time=2-00:00:00 --gres=gpu:1 --qos=gpu_access --partition=dschridelab --constraint=rhel8 --wrap "export LD_LIBRARY_PATH=~/.local/lib:$LD_LIBRARY_PATH && python3 src/models/train.py --config {0} --ifile {1} --odir {2} --tag {3}"'
    
    for ifile in ifiles:
        tag = ifile.split('/')[-1].split('.')[0]
        
        cmd_ = cmd.format(args.config, ifile, args.odir, tag)
        print(cmd_)
        
        os.system(cmd_)

if __name__ == '__main__':
    main()

