# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
import os
import argparse
import logging

import itertools
import pickle
import random


# use this format to tell the parsers
# where to insert certain parts of the script
# ${imports}

def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser()
    # my args
    parser.add_argument("--verbose", action = "store_true", help = "display messages")
    parser.add_argument("--idir", default = "None")
    parser.add_argument("--ofile", default = "None")

    parser.add_argument("--val_prop", default = "0.05")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
        logging.debug("running in verbose mode")
    else:
        logging.basicConfig(level=logging.INFO)



    return args

def main():
    args = parse_args()
    
    ifiles = os.listdir(args.idir)
    random.shuffle(ifiles)
    
    n_val = int(len(ifiles) * float(args.val_prop))
    
    val = ifiles[:n_val]
    del ifiles[:n_val]
    
    pickle.dump([ifiles, val], open(args.ofile, 'wb'))        

    # ${code_blocks}

if __name__ == '__main__':
    main()


