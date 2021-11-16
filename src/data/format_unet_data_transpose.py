# -*- coding: utf-8 -*-
import os
import argparse
import logging

import glob
import numpy as np

from data_functions import load_data

from mpi4py import MPI
import h5py

from seriate import seriate
from scipy.spatial.distance import pdist, cdist

from scipy.sparse.linalg import eigs
from sklearn.metrics import pairwise_distances
from scipy.optimize import linear_sum_assignment

import matplotlib.pyplot as plt
from sparsenn.models.gcn.topologies import knn_1d, random_graph_1d

from sklearn.neighbors import kneighbors_graph


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

def remove_singletons(x_list, y_list):
    
    if not len(x_list) == len(y_list):
        
        raise ValueError("training and truth data must have same number of chunks")
        
    new_x, new_y = [], []
        
    for idx, x in enumerate(x_list):
        
        y = y_list[idx]
        
        flt = (np.sum(x, axis=0) > 1)
        
        x_flt = x[:, flt]
        
        y_flt = y[:, flt]
        
        new_x.append(x_flt)
        
        new_y.append(y_flt)
        
    return new_x, new_y

class Formatter(object):
    def __init__(self, shape = (2, 128, 256), pop_sizes = [150, 156], 
                 sort_pops = True, sort_pos = False, ix_y = 1, metric = 'cosine'):
        
        self.n_pops = shape[0]
        self.pop_size = shape[1]
        self.n_sites = shape[2]
        
        self.pop_sizes = pop_sizes
        self.sort = sort_pops
        self.sort_pos = sort_pos
        
        self.ix_y = ix_y
        self.metric = metric
        
    # return a list of the desired array shapes
    def format(self, x, y):
        x1_indices = list(range(self.pop_sizes[0]))
        x2_indices = list(range(self.pop_sizes[0], self.pop_sizes[0] + self.pop_sizes[1]))
        
        x1 = x[x1_indices,:]
        x2 = x[x2_indices,:]
        
        y1 = y[x1_indices,:]
        y2 = y[x2_indices,:]
        
        if self.sort:
            x1, ix1 = seriate_spectral(x1)
            
            y1 = y1[ix1,:]
            
            C = cdist(x1, x2, metric = self.metric).astype(np.float32)
            i, j = linear_sum_assignment(C)
            
            y2 = y2[j, :]
            
        
        x = np.vstack([x1, x2])
        
        if self.sort_pos:
            x, ix = seriate_spectral(x.T)
            x = x.T
        else:
            ix = list(range(x.shape[1]))
        
        if self.ix_y == 0:
            return x, y1[:, ix]
        elif self.ix_y == 1:
            return x, y2[:, ix]
        else:
            return x, np.vstack([y1, y2])[:, ix]

        
            
            
def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser()
    # my args
    parser.add_argument("--verbose", action = "store_true", help = "display messages")
    parser.add_argument("--idir", default = "None")
    parser.add_argument("--chunk_size", default = "4")
    parser.add_argument("--istem", default = "None")

    parser.add_argument("--odir", default = "None")
    
    parser.add_argument("--pop_sizes", default = "150,156")
    parser.add_argument("--k", default = "5")
    parser.add_argument("--n_sites", default = "512")
    
    parser.add_argument("--densify", action = "store_true")
    parser.add_argument("--topology", default = "knn")
    
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
    ancFiles = [u.replace('txt', 'anc') for u in msFiles]

    ## patch for old naming system, to fix
    if len(msFiles) == 0:
        msFiles = [os.path.join(args.idir, 'mig.msOut')]
        ancFiles = [os.path.join(args.idir, 'out.anc')]
        
    n_sites = int(args.n_sites)
    k_neighbors = int(args.k)
    
    counter = 0
    for ix in range(len(msFiles)):
        n_received = 0
        
        msFile = msFiles[ix]
        ancFile = ancFiles[ix]
        
        comm.Barrier()
        
        x, y, positions = load_data(msFile, ancFile)
        
        if args.densify:
            x, y = remove_singletons(x, y)
        
        n_expected = len(x)
        
        pop_sizes = list(map(int, args.pop_sizes.split(',')))
        pop_size = sum(pop_sizes)
        
        if comm.rank != 0:            
            for ix in range(comm.rank - 1, len(x), comm.size - 1):
                x_ = x[ix]
                y_ = y[ix]
                pos = positions[ix]
                
                logging.info('{},{}'.format(x_.shape, y_.shape))
                if x_.shape[0] != pop_size:
                    logging.info('pop size error (x) in sim {0} in msFile {1}'.format(ix, msFile))
                    
                    comm.send([None, None, None], dest = 0)
                    continue
                elif y_.shape[0] != pop_size:
                    logging.info('pop size error (y) in sim {0} in msFile {1}'.format(ix, msFile))
                    
                    comm.send([None, None, None], dest = 0)
                    continue
                elif x_.shape[1] != y_.shape[1]:
                    logging.info('seg site mismatch error in sim {0} in msFile {1}'.format(ix, msFile))
                    
                    comm.send([None, None, None], dest = 0)
                    continue
                    
                ii = np.random.choice(range(x_.shape[1] - n_sites))
                
                x_ = x_[:, ii:ii + n_sites]
                y_ = y_[:, ii:ii + n_sites]
                
                f = Formatter(ix_y = int(args.ix_y), pop_sizes = pop_sizes)
                x_, y_ = f.format(x_, y_)
            
                comm.send([x_, y_, pos], dest = 0)
                
            del x
            del y
        else:
            del x
            del y
            
            while n_received < n_expected:
                x_, y, p = comm.recv(source = MPI.ANY_SOURCE)
                
                if x_ is not None:
                    n = x_.shape[0]
                    
                    co = []
                    for ix in range(pop_sizes[0]):
                        co.append((0, ix))
                        
                    for ix in range(pop_sizes[1]):
                        co.append((1, ix))
                        
                    co = np.array(co, dtype = np.float32)
                    
                    A = kneighbors_graph(co, k_neighbors, mode='connectivity', include_self = False).tocoo()
                    edge_index = np.array([A.row, A.col])
            
                    np.savez(os.path.join(args.odir, '{0:06d}.npz'.format(counter)), x = x_, y = y, edge_index = edge_index, pos = p)
                
                    counter += 1
                    
                n_received += 1
    
                if (n_received + 1) % 100 == 0:
                    logging.info('on {}...'.format(n_received))
                    
        comm.Barrier()
        
if __name__ == '__main__':
    main()