# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import logging
import argparse
import os

import torch

import h5py

import sys
sys.path.insert(0, os.path.join(os.getcwd(), 'src/models'))

from layers import NestedUNet
from data_loaders import H5UDataGenerator

from scipy.special import expit

def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser()
    # my args
    parser.add_argument("--verbose", action = "store_true", help = "display messages")
    parser.add_argument("--weights", default = "None") # weights of the pre-trained model
    parser.add_argument("--ifile", default = "None")
    
    parser.add_argument("--odir", default = "None")
    parser.add_argument("--n_samples", default = "4")
    
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
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    model = NestedUNet(1, 2)
    if args.weights != "None":
        checkpoint = torch.load(args.weights, map_location = device)
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()
    
    generator = H5UDataGenerator(h5py.File(args.ifile, 'r'), batch_size = 4)
    counter = 0
    
    for ix in range(int(args.n_samples)):
        with torch.no_grad():
            x, y = generator.get_batch()
            
            x = x.to(device)
            y = y.to(device)

            y_pred = model(x)
            
            x = x.detach().cpu().numpy()
            y = y.detach().cpu().numpy()
            y_pred = y_pred.detach().cpu().numpy()
            
            for k in range(x.shape[0]):
                fig, axes = plt.subplots(nrows = 5)
                axes[0].imshow(x[k,0,:,:], cmap = 'gray')
                axes[0].set_title('pop A')
                
                axes[1].imshow(x[k,1,:,:], cmap = 'gray')
                axes[1].set_title('pop B')
                
                axes[2].imshow(y[k,0,:,:], cmap = 'gray')
                axes[2].set_title('pop B (y)')
                
                axes[3].imshow(np.round(expit(y_pred[k,0,:,:])), cmap = 'gray')
                axes[3].set_title('pop B (pred)')
                
                im = axes[4].imshow(expit(y_pred[k,0,:,:]))
                axes[4].set_title('pop B (pred prob)')
                fig.colorbar(im, ax = axes[4])
                
                plt.savefig(os.path.join(args.odir, '{0:04d}_pred.png'.format(counter)), dpi = 100)
                counter += 1

    
if __name__ == '__main__':
    main()