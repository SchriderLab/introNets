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
from scipy.spatial.distance import pdist

from scipy.sparse.linalg import eigs
from sklearn.metrics import pairwise_distances

import matplotlib.pyplot as plt

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

def seriate_y_spectral(x):
    x = x.T
    
    C = pairwise_distances(x)
    C[np.where(np.isnan(C))] = 0.

    C = np.diag(C.sum(axis = 1)) - C
    _, v = eigs(C, k = 2, which = 'SM')

    f = v[:,1]
    ix = np.argsort(np.real(f))

    x = x[ix,:]
    
    return x.T, ix

class Formatter(object):
    def __init__(self, x, y, shape = (2, 128, 256), pop_sizes = [150, 156], sorting = None):
        # list of x and y arrays
        self.x = x
        self.y = y
        
        self.n_pops = shape[0]
        self.pop_size = shape[1]
        self.n_sites = shape[2]
        
        self.pop_sizes = pop_sizes
        self.sorting = sorting
        
    # return a list of the desired array shapes
    def format(self, x, y):
        x1_indices = list(np.random.choice(range(self.pop_sizes[0]), self.pop_size))
        x2_indices = list(np.random.choice(range(self.pop_sizes[0], self.pop_sizes[0] + self.pop_sizes[1]), self.pop_size))
        

        
        x1 = x[x1_indices, :]
        x2 = x[x2_indices, :]
        
        y1 = y[x1_indices, :]
        y2 = y[x2_indices, :]
        
        x1, i1 = seriate_x(x1)
        x2, i2 = seriate_x(x2)
    
        y1 = y1[i1, :]
        y2 = y2[i2, :]
        
        y = np.vstack([y1, y2])
        x = np.vstack([x1, x2])
        
        x, ii = seriate_y_spectral(x)
        
        x1 = x[:128, :]
        x2 = x[128:, :]
        
        y = y[:,ii]
        y1 = y[:128, :]
        y2 = y[128:, :]
            
        ii = np.where(x.sum(axis = 0) > 0)
        x1 = np.squeeze(x1[:,ii])
        x2 = np.squeeze(x2[:,ii])
        y2 = np.squeeze(y2[:,ii])
        
        six = np.random.choice(range(x1.shape[1] - self.n_sites))
        
        x = np.array([x1[:,six:six + self.n_sites], x2[:, six:six + self.n_sites]])
        #y = np.array([y1[:,six:six + self.n_sites], y2[:, six:six + self.n_sites]])
        y2 = y2[:, six:six + self.n_sites]
            
        return x, y2
            
            
def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser()
    # my args
    parser.add_argument("--verbose", action = "store_true", help = "display messages")
    parser.add_argument("--idir", default = "None")
    parser.add_argument("--chunk_size", default = "4")

    parser.add_argument("--ofile", default = "None")
    parser.add_argument("--sorting", default = "None")
    
    parser.add_argument("--scenario", default = "BF_to_AO")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
        logging.debug("running in verbose mode")
    else:
        logging.basicConfig(level=logging.INFO)

    return args

def main():
    args = parse_args()
    
    # configure MPI
    comm = MPI.COMM_WORLD
    
    if comm.rank == 0:
        ofile = h5py.File(args.ofile, 'w')

    idirs = [u for u in sorted(glob.glob(os.path.join(args.idir, '*/out*/out*'))) if ((not '.' in u) and (args.scenario in u))]
    chunk_size = int(args.chunk_size)

    if comm.rank != 0:
        for idir in idirs:
            msFile = os.path.join(idir, '{}.txt'.format(idir.split('/')[-1]))
            ancFile = os.path.join(idir, 'out.anc')
            
            x, y = load_data(msFile, ancFile)
            
            comm.Barrier()
            
            for ix in range(comm.rank - 1, len(x), comm.size - 1):
                x_ = x[ix]
                y_ = y[ix]
                
                if x_.shape != y_.shape:
                    comm.send([None, None], dest = 0)
                    continue
                
                f = Formatter(sorting = args.sorting)
                x_, y_ = f.format(x_, y_)
                
                comm.send([x, y], dest = 0)
                
    else:
        n_received = 0
        current_chunk = 0

        X = []
        Y = []
        
        while n_received < len(idirs) * 250:
            x, y = comm.recv(source = MPI.ANY_SOURCE)
            
            if x is not None:
                X.append(x)
                Y.append(y)
            else:
                n_received += 1
                continue
            
            n_received += 1
            
            while len(X) > chunk_size:
                ofile.create_dataset('{0}/x_0'.format(current_chunk), data = np.array(X[-chunk_size:], dtype = np.uint8), compression = 'lzf')
                ofile.create_dataset('{0}/y'.format(current_chunk), data = np.array(Y[-chunk_size:], dtype = np.uint8), compression = 'lzf')
                ofile.flush()
                
                del X[-chunk_size:]
                del Y[-chunk_size:]

                logging.info('0: wrote chunk {0}'.format(current_chunk))
                
                current_chunk += 1
                    

        ofile.close()
            
    # ${code_blocks}

if __name__ == '__main__':
    main()
# -*- coding: utf-8 -*-

