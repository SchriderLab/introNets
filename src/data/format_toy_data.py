# -*- coding: utf-8 -*-
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

from format_unet_data_h5 import Formatter, remove_singletons
from data_functions import load_data_slim
            
def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser()
    # my args
    parser.add_argument("--verbose", action = "store_true", help = "display messages")
    parser.add_argument("--idir", default = "None")
    parser.add_argument("--chunk_size", default = "4")

    parser.add_argument("--ofile", default = "None")
    parser.add_argument("--sorting", default = "seriate_match")
    
    parser.add_argument("--pop_sizes", default = "32,32")
    parser.add_argument("--out_shape", default = "2,64,64")
    
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

    ms_files = sorted(glob.glob(os.path.join(args.idir, '*.ms.gz')))
    log_files = sorted(glob.glob(os.path.join(args.idir, '*log.gz')))
    
    chunk_size = int(args.chunk_size)
    
    pop_sizes = tuple(list(map(int, args.pop_sizes.split(','))))
    out_shape = tuple(list(map(int, args.out_shape.split(','))))
    
    n_ind = sum(pop_sizes)

    if comm.rank != 0:
        for ix in range(comm.rank - 1, len(ms_files), comm.size - 1):
            logging.info('{0}: on {1}...'.format(comm.rank, ix))
            
            ms = ms_files[ix]
            log = log_files[ix]
            
            x, _, y, _ = load_data_slim(ms, log, n_ind)
                
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
        
        while n_received < len(ms_files):
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

