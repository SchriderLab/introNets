# -*- coding: utf-8 -*-

import os
import argparse
import logging

import matplotlib.pyplot as plt
import pandas as pd


# use this format to tell the parsers
# where to insert certain parts of the script
# ${imports}

def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser()
    # my args
    parser.add_argument("--verbose", action = "store_true", help = "display messages")
    parser.add_argument("--ifile", default = "None")

    parser.add_argument("--odir", default = "block_plots")
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

    df = pd.read_csv(args.ifile, index_col = False)
    
    labels = [u for u in list(df.keys()) if u != 'ratio']
    
    for label in labels:
        y = df['ratio']
        x = df[label]
        
        plt.scatter(x, y)
        plt.savefig(os.path.join(args.odir, '{}.png'.format(label)), dpi = 100)
        
        plt.close()
        
    

if __name__ == '__main__':
    main()
