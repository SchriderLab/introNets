# start
import os
import argparse
import logging

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np

import torch

# use this format to tell the parsers
# where to insert certain parts of the script
# ${imports}

def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser()
    # my args
    parser.add_argument("--verbose", action = "store_true", help = "display messages")
    parser.add_argument("--weights", default = "None")

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
    
    c = torch.load(args.weights, map_location=torch.device('cpu'))
    
    keys = list(c.keys())
    
    layers = [u for u in keys if 'res.layers' in u]
    numbers = sorted(list(set([int(u.split('.')[2]) for u in layers])))
    
    print([(c[u].shape, u) for u in keys])
    
    for k in range(len(numbers)):
        fig, axes = plt.subplots(ncols = 2)
        
        key = 'res.layers.{}.nns.0.lin_l.weight'.format(k)
        
        wl = c[key].numpy()
        
        key = 'res.layers.{}.nns.0.lin_r.weight'.format(k)
        
        wr = c[key].numpy()
    
        axes[0].imshow(wl)
        axes[1].imshow(wr)
        
        plt.savefig('test{0:02d}.png'.format(k), dpi = 100)
        plt.close()
    

    # ${code_blocks}

if __name__ == '__main__':
    main()
