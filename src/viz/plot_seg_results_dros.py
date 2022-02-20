# -*- coding: utf-8 -*-
import os
import argparse
import logging

import matplotlib.pyplot as plt
import numpy as np

import pickle

# use this format to tell the parsers
# where to insert certain parts of the script
# ${imports}

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser()
    # my args
    parser.add_argument("--verbose", action = "store_true", help = "display messages")
    parser.add_argument("--ifile", default = "None") # precomputed segmentation prediction
    parser.add_argument("--chrom", default = "None") # original data
    parser.add_argument("--disc_pred", default = "None") # discriminator predictions

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

    ifile = np.load(args.chrom)
    pop1_x = ifile['simMatrix'].T
    pop2_x = ifile['sechMatrix'].T
    
    positions = ifile['positions']
    
    X = np.vstack([pop1_x, pop2_x]).astype(np.uint8)
    Y = np.load(args.ifile)['Y']

    i2 = list(np.load(args.ifile)['x2i'])
    i2_ = list(set(list(i2)))
    n = len(i2_)
    
    disc = pickle.load(open(args.disc_pred, 'rb'))
    
    ar = 18
    ix = 0
    for k in range(len(disc[0])):
        if disc[0][k][1] > 0.5:
            ii = disc[1][k]
            pos = positions[ii]
            if pos[0] > 4000000 and pos[-1] < 5000000:
                ci = chunks(ii, 256)
                
                for ii in ci:
                    pos = positions[ii]
                    
                    pred = Y[:n, ii]
                    pred_im = np.zeros(pred.shape)
                    pred_im[np.where(pred >= 0.75)] = 1
    
                    pop2_im = np.zeros(pop2_x[:,ii].shape + (3,), dtype = np.uint8)
        
                    # make introgressed zero alleles blue
                    pop2_im[np.where((pred_im == 1) & (pop2_x[:,ii] == 0))] = np.array([0, 0, 255])
                    pop2_im[np.where((pred_im == 1) & (pop2_x[:,ii] == 1))] = np.array([255, 0, 0])
                    pop2_im[np.where((pred_im == 0) & (pop2_x[:,ii] == 1))] = np.array([255, 255, 255])
                    
                    fig, axes = plt.subplots(nrows = 4)
                    fig.set_size_inches(16, 6)
                    
                    axes[0].imshow(pop1_x[:,ii], cmap = 'gray', extent = (pos[0],pos[-1], pop1_x.shape[0], 0))
                    axes[0].set_aspect(ar)
                    
                    axes[1].imshow(pop2_im, extent = (pos[0], pos[-1], pop2_x.shape[0], 0))
                    axes[1].set_aspect(ar)
                    
                    pop2_im = np.zeros(pop2_x.shape + (3,))
                    
                    im = axes[2].imshow(pred, extent = (pos[0],pos[-1], n, 0))
                    fig.colorbar(im, ax = axes[2])
                    axes[2].set_aspect(ar)
                    
                    axes[3].imshow(pred_im, cmap = 'gray', extent = (pos[0], pos[-1], n, 0))
                    axes[3].set_aspect(ar)            
        
                    plt.savefig(os.path.join(args.odir, '{0:08d}_{1:08d}.png'.format(pos[0], pos[-1])), dpi = 100)
                    plt.close()
    
            

if __name__ == '__main__':
    main()

