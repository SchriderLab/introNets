# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
import os
import argparse
import logging

import itertools
import pickle


# use this format to tell the parsers
# where to insert certain parts of the script
# ${imports}

def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser()
    # my args
    parser.add_argument("--verbose", action = "store_true", help = "display messages")
    parser.add_argument("--idir", default = "None")
    parser.add_argument("--indices", default = "None")

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
    
    idirs = [os.path.join(args.idir, u) for u in os.listdir(args.idir)]
    
    cmd = 'sbatch --job-name=pytorch-ac --ntasks=1 --cpus-per-task=1 --mem=64G --time=2-00:00:00 --partition=volta-gpu --gres=gpu:1 --qos=gpu_access --wrap "export LD_LIBRARY_PATH=~/.local/lib:$LD_LIBRARY_PATH && python3 src/models/train_gcn_unet.py --idir {0} --odir {1} --tag {2} --indices {3}--layer_type gat --batch_size 12"'
    
    for idir in idirs:
        tag = idir.split('/')[-1]
        cmd_ = cmd.format(idir, args.odir, tag, args.indices)
        
        print(cmd_)
        os.system(cmd_)
        
    # ${code_blocks}

if __name__ == '__main__':
    main()


