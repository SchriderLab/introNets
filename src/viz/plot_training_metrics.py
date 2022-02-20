# -*- coding: utf-8 -*-
import os
import argparse
import logging

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# use this format to tell the parsers
# where to insert certain parts of the script
# ${imports}

def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser()
    # my args
    parser.add_argument("--verbose", action = "store_true", help = "display messages")
    parser.add_argument("--idir", default = "None")
    parser.add_argument("--ofile", default = "plots/1a_curves.png")
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

    ifiles = [os.path.join(args.idir, u) for u in os.listdir(args.idir) if (u.split('.')[-1] == 'csv' and 'seg' in u)]
    
    fig, axes = plt.subplots(ncols = 2, sharey = True)
    
    lines = []
    labels = []
    
    val_lines = []
    
    for ifile in ifiles:
        tag = ifile.split('/')[-1].split('.')[0].replace('_history', '')
        
        df = pd.read_csv(ifile)
        loss = np.array(df['loss'])
        val_loss = np.array(df['val_loss'])
        
        l1, = axes[0].plot(loss)
        l2, = axes[1].plot(val_loss)
        
        lines.append(l1)
        val_lines.append(l2)
        
        labels.append(tag)
        
    axes[0].legend(lines, labels, loc='upper right', shadow=True)
    axes[1].legend(val_lines, labels,  loc='upper right', shadow=True)
    
    plt.savefig(args.ofile, dpi = 100)
    plt.close()
        
if __name__ == '__main__':
    main()

