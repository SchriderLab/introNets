import os
import argparse
import logging

import h5py
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
from subprocess import Popen, PIPE
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from scipy.spatial.distance import squareform

# use this format to tell the parsers
# where to insert certain parts of the script
# ${imports}

import random
from scipy.sparse.linalg import eigs

from seriate import seriate

def seriate_spectral(C):    
    C = np.diag(C.sum(axis = 1)) - C
    _, v = eigs(C, k = 2, which = 'SM')

    f = v[:,1]
    ix = np.argsort(np.real(f))
    
    return ix

def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser()
    # my args
    parser.add_argument("--verbose", action = "store_true", help = "display messages")
    parser.add_argument("--ifile", default = "None")
    parser.add_argument("--n_samples", default = "5")

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

import copy

def main():
    args = parse_args()

    ifile = h5py.File(args.ifile, 'r')
    keys = list(ifile.keys())
    
    random.shuffle(keys)

    for ix in range(int(args.n_samples)):
        key = keys[ix]
        
        graph_keys = sorted(list(map(int, ifile[key]['graph'].keys())))
        
        cmd = "ffmpeg -y -f rawvideo -loglevel panic -vcodec rawvideo -s {1}x{2} -pix_fmt rgb24 -r 4 -i - -an -vcodec libx264 -pix_fmt yuv420p {0}".format(
                        os.path.join(args.odir, 'D_{0:03d}.mp4'.format(ix)), 1200, 600).split(' ')
        p = Popen(cmd, stdin=PIPE)
        
        bps = np.array(ifile[key]['break_points'])
        #print(bps, len(bps), len(graph_keys))
        
        y = np.array(ifile[key]['y'])[150:,:512]
        x = np.array(ifile[key]['x'])[150:,:512]
        
        fig = plt.figure(figsize = (8, 8))
        ax = plt.subplot(122)
        
        ax.imshow(y, cmap = 'gray')
        for p_ in bps[np.where(bps < 512)]:
            ax.plot([p_, p_], [0, y.shape[0] - 1], color = 'r')
        
        ax = plt.subplot(121)
        
        ax.imshow(x, cmap = 'gray')
        for p_ in bps[np.where(bps < 512)]:
            ax.plot([p_, p_], [0, x.shape[0] - 1], color = 'r')
        
        plt.savefig(os.path.join(args.odir, '{0:03d}_y.png'.format(ix)))
        plt.close()
            
        counter = 0
        start_snp = 0
        for gk in graph_keys:
            #print(gk)
            
            if 'D' not in list(ifile[key]['graph']['{}'.format(gk)].keys()):
                continue
            
            D = np.array(ifile[key]['graph']['{}'.format(gk)]['D'])
            D = squareform(D)
            
            ix1 = list(range(150))
            ix2 = list(range(150, 300))
            
            fig = plt.figure(figsize=(12, 6))
            ax = plt.subplot(132)
            ax.set_title('original order')
            
            im = ax.imshow(D)
            fig.colorbar(im, ax = ax)
            
            ax = plt.subplot(131)
            ax.set_title('seriated')
            
            ix_seriated = seriate(D, timeout = 0)
            D = D[ix_seriated,:]
            D = D[:,ix_seriated]
            
            ax.imshow(D)
            
            ax = plt.subplot(133)
            end_snp = bps[counter]
            counter += 1
            
            ax.imshow(y[:,start_snp:end_snp], cmap = 'gray', extent = (0, 1, 0, 1))
            
            start_snp = copy.copy(end_snp)
            
            canvas = FigureCanvas(fig)
            canvas.draw()

            buf = np.fromstring(canvas.tostring_rgb(), dtype='uint8').reshape(600, 1200, 3)
            p.stdin.write(buf.tostring())

            plt.close()

        p.stdin.close()
        p.wait()
        
        
            
            
            
            
            

    # ${code_blocks}

if __name__ == '__main__':
    main()
