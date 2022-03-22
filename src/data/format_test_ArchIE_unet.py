# -*- coding: utf-8 -*-
import os
import argparse
import logging

import os
import numpy as np
import logging, argparse

from data_functions import load_data, get_windows

import h5py

import matplotlib.pyplot as plt
from format_unet_data_h5 import Formatter
from collections import deque

from calc_stats_ms import *
from mpi4py import MPI

# use this format to tell the parsers
# where to insert certain parts of the script
# ${imports}

def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser()
    # my args
    parser.add_argument("--verbose", action = "store_true", help = "display messages")
    parser.add_argument("--idir", default = "None")
    
    parser.add_argument("--pop_sizes", default = "100,100")
    parser.add_argument("--out_shape", default = "2,104,128")
    
    parser.add_argument("--step_size", default = "1")
    parser.add_argument("--n_per_dir", default = "100")

    parser.add_argument("--ofile", default = "None")
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

    idirs = [os.path.join(args.idir, u) for u in os.listdir(args.idir) if not '.' in u]
    idirs_ = []
    for idir in idirs:
        ms = os.path.join(idir, 'mig.msOut')
        anc = os.path.join(idir, 'out.anc')
        
        if os.path.exists(ms) and os.path.exists(anc):
            idirs_.append(idir)
    
    idirs = idirs_
    
    pop_sizes = tuple(list(map(int, args.pop_sizes.split(','))))
    out_shape = tuple(list(map(int, args.out_shape.split(','))))
    
    window_size = out_shape[-1]
    
    n_per_dir = int(args.n_per_dir)
    n_reps = n_per_dir * len(idirs)
    
    if comm.rank != 0:
        for ix in range(comm.rank - 1, len(idirs), comm.size - 1):
            logging.info('working on {}...'.format(idirs[ix]))
            
            ms = os.path.join(idirs[ix], 'mig.msOut')
            anc = os.path.join(idirs[ix], 'out.anc')

            X, Y, P = load_data(ms, anc, leave_out_last = True)
            X = deque(X)
            Y = deque(Y)
            P = deque(P)
            
            while len(X) > 0: 
                x = X.pop()
                y = Y.pop()
                p = p.pop()
                
                X_ = []
                Y_ = []
                indices = []
                positions = []
                for ix in range(x.shape[1] - window_size):
                    pi = range(ix,ix + window_size)
                    
                    x_ = x[:,ix:ix + window_size]
                    y_ = y[:,ix:ix + window_size]
                    
                    f = Formatter([x_], [y_], sorting = args.sorting, pop = 0, 
                                  pop_sizes = pop_sizes, shape = out_shape)
                    x_, y_, i1 = f.format(True)
                
                    X_.append(x_)
                    Y_.append(y_)
                    indices.append(i1)
                    positions.append(pi)
                    
                X_ = np.array(X_, dtype = np.uint8)
                Y_ = np.array(Y_, dtype = np.uint8)
                indices = np.array(indices, dtype = np.int32)
                positions = np.array(positions, dtype = np.int32)
                
                comm.send([X_, Y_, indices, positions], dest = 0)
                
    else:
        n_received = 0
        current_chunk = 0

        while n_received < n_reps:
            x, y, indices, pi = comm.recv(source = MPI.ANY_SOURCE)
            logging.info('writing set {}...'.format(current_chunk))
            
            ofile.create_dataset('{}/x_0'.format(current_chunk), data = x, compression = 'lzf')
            ofile.create_dataset('{}/y'.format(current_chunk), data = y, compression = 'lzf')
            ofile.create_dataset('{}/indices'.format(current_chunk), data = indices, compression = 'lzf')
            ofile.create_dataset('{}/ix'.format(current_chunk), data = pi, compression = 'lzf')
            
            current_chunk += 1
            n_received += 1

            ofile.flush()

        ofile.close()            
            
                
    
    

if __name__ == '__main__':
    main()
