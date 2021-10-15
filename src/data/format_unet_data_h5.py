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
from scipy.spatial.distance import pdist

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
    def __init__(self, x, y, shape = (2, 128, 128), pop_sizes = [150, 156], sorting = None, pop = None):
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
    def format(self):
        X = []
        Y = []
        
        for k in range(len(self.x)):
            x = self.x[k]
            y = self.y[k]
            
            x1_indices = list(np.random.choice(range(self.pop_sizes[0]), self.pop_size))
            x2_indices = list(np.random.choice(range(self.pop_sizes[0], self.pop_sizes[0] + self.pop_sizes[1]), self.pop_size))
            
            x1 = x[x1_indices, :]
            x2 = x[x2_indices, :]
            
            y1 = y[x1_indices, :]
            y2 = y[x2_indices, :]
            
            six = np.random.choice(range(x1.shape[1] - self.n_sites))
            
            x = np.array([x1[:,six:six + self.n_sites], x2[:, six:six + self.n_sites]])
            y = np.array([y1[:,six:six + self.n_sites], y2[:, six:six + self.n_sites]])
            
            if self.pop:
                if self.pop == "0":
                    y = y[0]
                else:
                    y = y[1]
            
            X.append(x)
            Y.append(y)
            
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
    
    parser.add_argument("--densify", action = "store_true", help = "remove singletons")
    
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

    idirs = [u for u in sorted(glob.glob(os.path.join(args.idir, '*/out*/out*'))) if (not '.' in u)]
    chunk_size = int(args.chunk_size)

    if comm.rank != 0:
        for ix in range(comm.rank - 1, len(idirs), comm.size - 1):
            logging.info('{0}: on {1}...'.format(comm.rank, ix))
            
            idir = idirs[ix]
            
            msFile = os.path.join(idir, '{}.txt'.format(idir.split('/')[-1]))
            ancFile = os.path.join(idir, 'out.anc')
            
            x, y = load_data(msFile, ancFile)
            
            if args.densify:
                x, y = remove_singletons(x, y)
            
            f = Formatter(x, y, sorting = args.sorting, pop = args.pop)
            x, y = f.format()
        
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
            

    # ${code_blocks}

if __name__ == '__main__':
    main()
