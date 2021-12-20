# -*- coding: utf-8 -*-
import os
import argparse
import logging

import torch
import h5py
import numpy as np

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
# use this format to tell the parsers
# where to insert certain parts of the script
# ${imports}

def calc(x, mean, std, beta = 8, interp = 4, device=torch.device('cuda')):
    """Calculate average power spectrum and store it in .npz file."""
    image_size = 128
    spectrum_size = image_size * interp
    padding = spectrum_size - image_size

    # Setup window function.
    window = torch.kaiser_window(image_size, periodic=False, beta=beta, device=device)[64]
    window *= window.square().sum().rsqrt()
    window = window.ger(window).unsqueeze(0)
    
    print(window.shape)

    x = (x.to(torch.float64) - mean) / std
    x = torch.nn.functional.pad(x * window, [0, padding])
    spectrum = torch.fft.fftn(x, dim=[2]).abs().square().mean(dim=[0,1])

    return spectrum

def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser()
    # my args
    parser.add_argument("--verbose", action = "store_true", help = "display messages")
    parser.add_argument("--ifile", default = "None")

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
    
    ifile = h5py.File(args.ifile, 'r')
    keys = list(ifile['train'].keys())
    device = torch.device('cuda')
    
    means = []
    variances = []
    
    logging.info('computing mean and variance...')
    for key in np.random.choice(keys, 1000, replace = False):
        x = np.array(ifile['train'][key]['x_0'])
        
        mean = np.mean(x)
        variance = np.var(x)
        
        means.append(mean)
        variances.append(variance)
    
    mean = np.mean(means)
    std = np.sqrt(np.mean(variances))

    logging.info('computing_spectras...')    
    n = 0
    spectrum_size = 4 * 128
    spectrum = torch.zeros(spectrum_size).to(torch.float64).to(device)
    for key in np.random.choice(keys, 1000, replace = False):
        x = np.array(ifile['train'][key]['x_0'])
        ix = list(range(64)) + list(range(150, 214))
        x = x[:,ix,:]
        x = np.expand_dims(x, 1)
        
        x = torch.FloatTensor(x).to(device)
    
        print(x.shape)
        
        s = calc(x, mean, std).squeeze(0)
        
        spectrum += s
        n += 1
        
    spectrum /= n
    spectrum = spectrum.detach().cpu().numpy()
    
    logging.info('plotting and saving...')
    np.savez(os.path.join(args.odir, 'average_spectrum.npz'), spectrum = spectrum)
        
    plt.rc('font', family = 'Helvetica', size = 12)
    plt.rcParams.update({'figure.autolayout': True})
    fig = plt.figure(figsize=(12, 8), dpi=100)
        
    ax = fig.add_subplot(111)
    ax.plot(np.linspace(0., 128, spectrum_size), spectrum)
    
    plt.savefig(os.path.join(args.odir, 'average_spectrum.png'), dpi = 100)
    plt.close()

    # ${code_blocks}

if __name__ == '__main__':
    main()

