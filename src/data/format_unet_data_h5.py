# -*- coding: utf-8 -*-
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

from scipy.optimize import linear_sum_assignment
from scipy.interpolate import interp1d

def make_continuous(x):
    x = np.cumsum(x, axis = 1) * 2 * np.pi

    mask = np.zeros(x.shape)
    ix = list(np.where(np.diff(x) != 0))
    ix[-1] += 1
    mask[tuple(ix)] = 1
    mask[:,-1] = 1
    
    x[mask == 0] = 0
    t = np.array(range(x.shape[-1]))
    
    for k in range(len(x)):
        ix = [0] + list(np.where(x[k] != 0)[0])
        print(len(ix))
        
        t = np.array(range(np.max(ix)))
        
        if len(ix) > 3:
            x[k,:len(t)] = interp1d(ix, x[k,ix], kind = 'cubic')(t)
        elif len(ix) > 2:
            x[k,:len(t)] = interp1d(ix, x[k,ix], kind = 'quadratic')(t)
        elif len(ix) > 1:
            x[k,:len(t)] = interp1d(ix, x[k,ix], kind = 'linear')(t)
            
    x = np.cos(x)
    return x

def seriate_x(x):
    Dx = pdist(x, metric = 'cosine')
    Dx[np.where(np.isnan(Dx))] = 0.
    ix = seriate(Dx)

    return x[ix], ix
    
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
    def __init__(self, x, y, shape = (2, 32, 64), pop_sizes = [150, 156], sorting = None, pop = None):
        # list of x and y arrays
        self.x = x
        self.y = y
        
        self.n_pops = shape[0]
        self.pop_size = shape[1]
        self.n_sites = shape[2]
        
        self.pop_sizes = pop_sizes
        self.sorting = sorting
        self.pop = pop
        
    # return a list of the desired array shapes
    def format(self, return_indices = False, continuous = False, zero = False):
        X = []
        Y = []
        
        for k in range(len(self.x)):
            x = self.x[k]
            
            if self.y is not None:
                y = self.y[k]
            
            #print(x.shape, y.shape)
            
            if not return_indices:
                if x.shape[0] != sum(self.pop_sizes) or y.shape[0] != sum(self.pop_sizes):
                    continue
                
                # upsample the populations if need be
                x1_indices = list(range(self.pop_sizes[0]))
                n = self.pop_size - self.pop_sizes[0]
                
                if n > self.pop_sizes[0]:
                    replace = True
                else:
                    replace = False
                
                if n > 0:
                    x1_indices = x1_indices + list(np.random.choice(range(self.pop_sizes[0]), n, replace = replace))
                
                # upsample the second pop (again if needed)
                x2_indices = list(range(self.pop_sizes[0], self.pop_sizes[0] + self.pop_sizes[1]))
                n = self.pop_size - self.pop_sizes[1]
                
                if n > self.pop_sizes[1]:
                    replace = True
                else:
                    replace = False
                
                if n > 0:
                    x2_indices = x2_indices + list(np.random.choice(range(self.pop_sizes[0], self.pop_sizes[0] + self.pop_sizes[1]), n, replace = replace))
                
                x1 = x[x1_indices, :]
                x2 = x[x2_indices, :]
            else:
                x1 = x[:self.pop_size,:]
                x2 = x[self.pop_size:,:]
                
                x1_indices = list(range(self.pop_size))
                x2_indices = list(range(self.pop_size))
            
            if continuous:
                x1 = make_continuous(x1)
                x2 = make_continuous(x2)
            
            if self.y is not None:
                y1 = y[x1_indices, :]
                y2 = y[x2_indices, :]
             
                if not zero:
                    indices = list(set(range(x1.shape[1] - self.n_sites)).intersection(list(np.where(np.sum(y2, axis = 0) > 0)[0])))
                    if len(indices) == 0:
                        continue
                else:
                    indices = list(range(x1.shape[1] - self.n_sites))
                
                six = np.random.choice(indices)
                
                if not zero:
                    if np.sum(y[:,six:six + self.n_sites]) == 0:
                        continue
                
                x1 = x1[:,six:six + self.n_sites]
                x2 = x2[:,six:six + self.n_sites]

                y1 = y1[:,six:six + self.n_sites]
                y2 = y2[:,six:six + self.n_sites]
            
            if self.sorting == "seriate_match":
                x1, ix1 = seriate_x(x1)
                
                D = cdist(x1, x2, metric = 'cosine')
                D[np.where(np.isnan(D))] = 0.
                
                i, j = linear_sum_assignment(D)
                
                x2 = x2[j,:]
                x2_indices = [x2_indices[u] for u in j]
                
                if self.y is not None:
                    y1 = y1[ix1, :]
                    y2 = y2[j, :]
            
            x = np.array([x1, x2])
            
            if self.y is not None:
                y = np.array([y1, y2])
            
            if self.y is not None:
                if self.pop:
                    if self.pop == "0":
                        y = np.expand_dims(y[0])
                    else:
                        y = np.expand_dims(y[1])
                        
                Y.append(y)
            
            X.append(x)

        if return_indices:
            return X[0], (x1_indices, x2_indices)
            
        return X, Y
            
            
def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser()
    # my args
    parser.add_argument("--verbose", action = "store_true", help = "display messages")
    parser.add_argument("--idir", default = "None")
    parser.add_argument("--chunk_size", default = "4")

    parser.add_argument("--ofile", default = "None")
    parser.add_argument("--sorting", default = "None")
    
    parser.add_argument("--pop_sizes", default = "150,156")
    parser.add_argument("--out_shape", default = "2,128,128")
    
    parser.add_argument("--densify", action = "store_true", help = "remove singletons")
    parser.add_argument("--zero", action = "store_true")
    
    parser.add_argument("--pop", choices = ["0", "1"], help = "only return y values for one pop?")

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

    idirs = [u for u in sorted(glob.glob(os.path.join(args.idir, '*')))]
    chunk_size = int(args.chunk_size)
    
    pop_sizes = tuple(list(map(int, args.pop_sizes.split(','))))
    out_shape = tuple(list(map(int, args.out_shape.split(','))))

    if comm.rank != 0:
        for ix in range(comm.rank - 1, len(idirs), comm.size - 1):
            logging.info('{0}: on {1}...'.format(comm.rank, ix))
            
            idir = idirs[ix]
            
            try:
                msFile = os.path.join(idir, '{}.txt'.format(idir.split('/')[-1]))
                ancFile = os.path.join(idir, '{}.anc'.format(idir.split('/')[-1]))
                
                x, y, _ = load_data(msFile, ancFile)
            except:
                msFile = os.path.join(idir, 'mig.msOut')
                ancFile = os.path.join(idir, 'out.anc')
                
                x, y, _ = load_data(msFile, ancFile)
                
            if args.densify:
                x, y = remove_singletons(x, y)
            
            f = Formatter(x, y, sorting = args.sorting, pop = args.pop, 
                          pop_sizes = pop_sizes, shape = out_shape)
            x, y = f.format(zero = args.zero)
        
            comm.send([x, y], dest = 0)
            
    else:
        n_received = 0
        current_chunk = 0

        X = []
        Y = []
        
        while n_received < len(idirs):
            x, y = comm.recv(source = MPI.ANY_SOURCE)
            
            X.extend(x)
            Y.extend(y)
            
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

if __name__ == '__main__':
    main()
