import os
import argparse
import logging

import glob
import numpy as np

from data_functions import load_data

from mpi4py import MPI
import h5py

# use this format to tell the parsers
# where to insert certain parts of the script
# ${imports}

from seriate import seriate
from scipy.spatial.distance import pdist, cdist

from scipy.sparse.linalg import eigs
from sklearn.metrics import pairwise_distances
from scipy.optimize import linear_sum_assignment

import matplotlib.pyplot as plt
from sparsenn.models.gcn.topologies import knn_1d

def seriate_x(x):
    Dx = pdist(x, metric = 'cosine')
    Dx[np.where(np.isnan(Dx))] = 0.
    ix = seriate(Dx, timeout = 0)

    return x[ix], ix

def seriate_y(x):
    x = x.T
    
    Dx = pdist(x, metric = 'cosine')
    Dx[np.where(np.isnan(Dx))] = 0.
    ix = seriate(Dx, timeout = 0)

    return x[ix].T, ix

def seriate_spectral(x):    
    C = pairwise_distances(x)
    C[np.where(np.isnan(C))] = 0.

    C = np.diag(C.sum(axis = 1)) - C
    _, v = eigs(C, k = 2, which = 'SM')

    f = v[:,1]
    ix = np.argsort(np.real(f))

    x = x[ix,:]
    
    return x, ix

class Formatter(object):
    def __init__(self, shape = (2, 128, 256), pop_sizes = [150, 156], 
                 sort = True, ix_y = 1, metric = 'cosine'):
        
        self.n_pops = shape[0]
        self.pop_size = shape[1]
        self.n_sites = shape[2]
        
        self.pop_sizes = pop_sizes
        self.sort = sort
        
        self.ix_y = ix_y
        self.metric = metric
        
    # return a list of the desired array shapes
    def format(self, x, y):
        x1_indices = list(np.random.choice(range(self.pop_sizes[0]), self.pop_size))
        x2_indices = list(np.random.choice(range(self.pop_sizes[0], self.pop_sizes[0] + self.pop_sizes[1]), self.pop_size))
        
        x1 = x[x1_indices, :]
        x2 = x[x2_indices, :]
        
        y1 = y[x1_indices, :]
        y2 = y[x2_indices, :]
        
        if self.sort:
            x1, ix1 = seriate_spectral(x1)
            
            y1 = y1[ix1, :]
            
            C = cdist(x1, x2, metric = self.metric).astype(np.float32)
            i, j = linear_sum_assignment(C)
            
            x2 = x2[j, :]
            y2 = y2[j, :]
            
        x = np.vstack([x1, x2])

        if self.ix_y == 1:
            return x, y2
        elif self.ix_y == 0:
            return x, y1
        elif self.ix_y == -1:
            return x, np.vstack([y1, y2])
        
            
            
def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser()
    # my args
    parser.add_argument("--verbose", action = "store_true", help = "display messages")
    parser.add_argument("--idir", default = "None")
    parser.add_argument("--chunk_size", default = "4")

    parser.add_argument("--odir", default = "None")
    
    parser.add_argument("--k", default = "16")
    parser.add_argument("--n_dilations", default = "7")
    
    parser.add_argument("--ix_y", default = "1")
    
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
        else:
            os.system('rm -rf {0}'.format(os.path.join(args.odir, '*')))


    return args

def main():
    # configure MPI
    comm = MPI.COMM_WORLD

    args = parse_args()

    chunk_size = int(args.chunk_size)
    
    msFiles = sorted(glob.glob(os.path.join(args.idir, '*.txt')))
    ancFiles = sorted(glob.glob(os.path.join(args.idir, '*.anc')))
    
    n_received = 0
    
    for ix in range(len(msFiles)):
        msFile = msFiles[ix]
        ancFile = ancFiles[ix]
        
        logging.info('loading files...')
        x, y = load_data(msFile, ancFile)
        
        comm.Barrier()
        
        if comm.rank != 0:
            for ix in range(comm.rank - 1, len(x), comm.size - 1):
                x_ = x[ix]
                y_ = y[ix]
                
                logging.info('{},{}'.format(x_.shape, y_.shape))
                
                f = Formatter(ix_y = int(args.ix_y))
                x_, y_ = f.format(x_, y_)
            
                comm.send([x_, y_], dest = 0)
        else:
            while n_received < len(x):
                x_, y = comm.recv(source = MPI.ANY_SOURCE)
                
                if x_ is not None:
                    n = x_.shape[0]
                
                    edges = [u.numpy() for u in knn_1d(n, k = int(args.k), n_dilations = int(args.n_dilations))]
                    
                    np.savez(os.path.join(args.odir, '{0:06d}.npz'.format(n_received)), x = x_, y = y, edges = edges)
                
                n_received += 1
    
                if (n_received + 1) % 100 == 0:
                    logging.info('on {}...'.format(n_received))
                    
        comm.Barrier()
        
    # ${code_blocks}

if __name__ == '__main__':
    main()
# -*- coding: utf-8 -*-

