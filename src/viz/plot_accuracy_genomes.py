import os
import argparse
import logging

import numpy as np

# use this format to tell the parsers
# where to insert certain parts of the script
# ${imports}

import glob
import pandas as pd

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser()
    # my args
    parser.add_argument("--verbose", action = "store_true", help = "display messages")
    parser.add_argument("--idir", default = "None")
    parser.add_argument("--ifile", default = "/proj/dschridelab/miscShared/toDylan/bgsTestRateInfo.txt")
    
    parser.add_argument("--ofile", default = "bgs_acc.png")
    parser.add_argument("--ofile_csv", default = "bgs_acc.csv")
    
    parser.add_argument("--prefix", default = "bgsSim")

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
    
    ifiles = sorted(glob.glob(os.path.join(args.idir, '*.npz')))
    
    df = pd.read_csv(args.ifile, sep = '\t')
    df = df.set_index('simFile')

    result = dict()
    result['ix'] = []
    result['r_mid'] = []
    result['r'] = []
    result['acc'] = []
    result['acc_mid'] = []
    
    for ix, ifile in enumerate(ifiles):
        rec_rate_mid = df['avgRecRateCentral100kb'][args.prefix + '_{}.slim'.format(ix)]
        rec_rate = df['avgRecRateFull1Mb'][args.prefix + '_{}.slim'.format(ix)]
        
        y = np.load(ifile)['y_true']
        y_pred = np.round(np.load(ifile)['y_pred'])
        
        pos = np.load(ifile)['pos']

        ii = np.where((pos >= 450000) & (pos <= 550000))
        
        acc_m = np.mean(np.abs(y[:,:,ii] - y_pred[:,:,ii]))
        acc = np.mean(np.abs(y - y_pred))
        
        result['r_mid'].append(rec_rate_mid)
        result['r'].append(rec_rate)
        result['acc_mid'].append(acc_m)
        result['acc'].append(acc)
        result['ix'].append(ix)

    fig, axes = plt.subplots(ncols = 2)
    axes[0].plot(result['r_mid'], result['acc_mid'])
    axes[1].plot(result['r'], result['acc'])

    plt.savefig(args.ofile, dpi = 100)
    plt.close()
    
    df = pd.DataFrame(result)
    df.to_csv(args.ofile_csv, index = False)
    

    # ${code_blocks}

if __name__ == '__main__':
    main()
