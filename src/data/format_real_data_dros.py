# -*- coding: utf-8 -*-
import os
import numpy as np
import logging, argparse

import h5py

import configparser
from mpi4py import MPI
import sys

from format_unet_data_h5 import Formatter

def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true", help="display messages")
    parser.add_argument("--ifile", default = "None")
    parser.add_argument("--ofile", default = "None")

    parser.add_argument("--chunk_size", default = "256")
    parser.add_argument("--format_config", default = "sorting_configs/drosophila_both.config")
    parser.add_argument("--no_sort", action = "store_true")
    parser.add_argument("--split", action = "store_true")
    
    parser.add_argument("--sorting", default = "seriate_match")
    
    parser.add_argument("--pop_sizes", default = "20,14")
    parser.add_argument("--out_shape", default = "2,32,64")
    
    parser.add_argument("--densify", action = "store_true", help = "remove singletons")

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
    
    pop_sizes = tuple(list(map(int, args.pop_sizes.split(','))))
    out_shape = tuple(list(map(int, args.out_shape.split(','))))
    
    pop_size = out_shape[1]  

    if comm.rank == 0:
        ifile = np.load(args.ifile)
        pop1_x = ifile['simMatrix'].T
        pop2_x = ifile['sechMatrix'].T
        
        # do the upsampling across all sites
        if pop_sizes[0] >= pop_size:
            replace = False
        else:
            replace = True
        
        x1_indices = list(np.random.choice(range(pop_sizes[0]), pop_size, replace = replace))
        
        if pop_sizes[1] >= pop_size:
            replace = False
        else:
            replace = True
        
        x2_indices = list(np.random.choice(range(pop_sizes[1]), pop_size, replace = replace))
        positions = ifile['positions']

        X = np.vstack((pop1_x[x1_indices,:], pop2_x[x2_indices,:]))
        
        shape = X.shape

        n_files = X.shape[1] - 64
        print(n_files)

    else:
        X = None
        positions = None

    comm.Barrier()

    X = comm.bcast(X, root=0)
    positions = comm.bcast(positions, root=0)

    comm.Barrier()

    chunk_size = int(args.chunk_size)

    if comm.rank != 0:
        for ix in range(comm.rank - 1, X.shape[1] - 64, comm.size - 1):
            x = X[:,ix:ix + 64]

            # destroy the perfect information regarding
            # which allele is the ancestral one
            for k in range(x.shape[1]):
                if np.sum(x[:, k]) > 17:
                    x[:, k] = 1 - x[:, k]
                elif np.sum(x[:, k]) == 17:
                    if np.random.choice([0, 1]) == 0:
                        x[:, k] = 1 - x[:, k]

            p = positions[ix:ix + 64]
            pi = range(ix, ix + 64)

            f = Formatter([x], None, sorting = args.sorting, pop = 0, 
                          pop_sizes = pop_sizes, shape = out_shape)
            x, i1 = f.format(True)

            comm.send([x, p, pi, i1], dest=0)

    else:
        X = None
        positions = None

        ofile = h5py.File(args.ofile, 'w')
        ofile.create_dataset('x1_indices', data = np.array(x1_indices, dtype = np.int32))
        ofile.create_dataset('x2_indices', data = np.array(x2_indices, dtype = np.int32))
        ofile.create_dataset('shape', data = np.array(shape, dtype = np.int32))

        n_received = 0
        current_chunk = 0

        X = []
        P = []
        Pi = []
        indices = []

        while n_received < n_files:
            x, p, pi, ix = comm.recv(source=MPI.ANY_SOURCE)

            n_received += 1

            X.append(x)
            P.append(p)
            Pi.append(pi)
            indices.append(ix)

            while len(X) >= chunk_size:
                ofile.create_dataset('{0}/x_0'.format(current_chunk), data = np.array(X[-chunk_size:], dtype = np.uint8), compression = 'lzf')
                ofile.create_dataset('{0}/positions'.format(current_chunk), data = np.array(P[-chunk_size:], dtype = np.int64), compression = 'lzf')
                ofile.create_dataset('{0}/pi'.format(current_chunk), data = np.array(Pi[-chunk_size:], dtype = np.int64), compression = 'lzf')

                if not args.split:
                    ofile.create_dataset('{0}/indices'.format(current_chunk), data = np.array(indices[-chunk_size:], dtype = np.uint8), compression = 'lzf')
                else:
                    ofile.create_dataset('{0}/i1'.format(current_chunk), data = np.array([u[0] for u in indices[-chunk_size:]], dtype = np.uint8), compression = 'lzf')
                    ofile.create_dataset('{0}/i2'.format(current_chunk), data = np.array([u[1] for u in indices[-chunk_size:]], dtype = np.uint8), compression = 'lzf')

                logging.info('0: wrote chunk {0}'.format(current_chunk))

                current_chunk += 1

                del X[-chunk_size:]
                del P[-chunk_size:]
                del Pi[-chunk_size:]
                del indices[-chunk_size:]

                ofile.flush()

        if len(X) > 0:
            ofile.create_dataset('{0}/x_0'.format(current_chunk), data=np.array(X, dtype=np.uint8),
                                 compression='lzf')
            ofile.create_dataset('{0}/positions'.format(current_chunk), data=np.array(P, dtype=np.int64),
                                 compression='lzf')
            ofile.create_dataset('{0}/pi'.format(current_chunk), data = np.array(Pi, dtype = np.int64), compression = 'lzf')
            
            if not args.split:
                ofile.create_dataset('{0}/indices'.format(current_chunk), data = np.array(indices[-chunk_size:], dtype = np.uint8), compression = 'lzf')
            else:
                ofile.create_dataset('{0}/i1'.format(current_chunk), data = np.array([u[0] for u in indices[-chunk_size:]], dtype = np.uint8), compression = 'lzf')
                ofile.create_dataset('{0}/i2'.format(current_chunk), data = np.array([u[1] for u in indices[-chunk_size:]], dtype = np.uint8), compression = 'lzf')

        ofile.close()
        
if __name__ == '__main__':
    main()
