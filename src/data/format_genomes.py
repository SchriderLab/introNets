# -*- coding: utf-8 -*-
import os
import argparse
import logging

import glob
import numpy as np

from data_functions import load_data, TwoPopAlignmentFormatter, load_data_slim, read_slim_out

from mpi4py import MPI
import h5py
import time
from collections import deque

# use this format to tell the parsers
# where to insert certain parts of the script
# ${imports}

def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser()
    # my args
    parser.add_argument("--verbose", action = "store_true", help = "display messages")
    parser.add_argument("--idir", default = "None", help = "input directory containing MS or SLiM simulations.  See simulate_msmodified.py and simulate_slim.py for how these simulations are formatted, organized etc.")
    parser.add_argument("--chunk_size", default = "4", help = "number of replicates per h5 key.  chunking data signficantly increases read speed (especially on traditional spinning hdds)")

    parser.add_argument("--ofile", default = "None", help = "hdf5 file to write to")
    parser.add_argument("--sorting", default = "seriate_match", help = "legacy option.  this or none are the only options implemented currently")
    parser.add_argument("--metric", default = "cosine", help = "metric to use when matching / seriating alignments")
    
    parser.add_argument("--region", default = "None", help = "specify a region of the simulated chromosomes to crop to before formatting further.  as a tuple a,b s.t. a,b in [0,1] and a > b")
    
    parser.add_argument("--pop_sizes", default = "64,64", help = "pop sizes.  we currently only support two-population scenarios")
    parser.add_argument("--out_shape", default = "2,128,128", help = "desired output shape.  channels (populations) x individuals x segregating sites")
    
    parser.add_argument("--densify", action = "store_true", help = "remove singletons")
    parser.add_argument("--include_zeros", action = "store_true", help = "used to include replicate windows that have no introgression")
    
    parser.add_argument("--pop", default = "0", help = "only return y values for one pop, pop not in [0, 1] == use both populations (bidirectional introgression case)")
    parser.add_argument("--step_size", default = "64")

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
    
    # configure MPI
    comm = MPI.COMM_WORLD
    
    if comm.rank == 0:
        ofile = h5py.File(args.ofile, 'w')

    ms_files = sorted(glob.glob(os.path.join(args.idir, '*.ms.gz')))
    anc_files = sorted(glob.glob(os.path.join(args.idir, '*.log.gz')))
    
    pop_sizes = tuple(list(map(int, args.pop_sizes.split(','))))
    out_shape = tuple(list(map(int, args.out_shape.split(','))))
    
    window_size = out_shape[-1]
    step_size = int(args.step_size)
    
    n_ind = sum(pop_sizes)
    
    if "," in args.region:
        region = tuple(map(float, args.region.split(',')))
    else:
        region = None
    
    if comm.rank != 0:
        for ix in range(comm.rank - 1, len(ms_files), comm.size - 1):
            logging.info('{0}: on {1}...'.format(comm.rank, ix))
            
            # are we formatting SLiM or MS data?  
            # auto-detected above
            # time the disk-read time
            t0 = time.time()
            
            X, P, Y = load_data_slim(ms_files[ix], anc_files[ix], n_ind, region = region)
            
            X = deque(X)
            Y = deque(Y)
            P = deque(P)

            while len(X) > 0:
                x = X.pop()
                y = Y.pop()
                logging.info('x_shape: {}'.format(x.shape))
                
                logging.info('have {} windows...'.format(len(range(0, x.shape[1] - window_size, step_size))))
                
                X_ = []
                Y_ = []
                indices = []
                positions = []
                for ij in range(0, x.shape[1] - window_size, step_size):
                    pi = list(range(ij,ij + window_size))
                    
                    x_ = x[:,ij:ij + window_size]
                    y_ = y[:,ij:ij + window_size]
                    
                    f = TwoPopAlignmentFormatter([x_], [y_], sorting = args.sorting, pop = int(args.pop), 
                                  pop_sizes = pop_sizes, shape = out_shape)
                    f.format(include_zeros = True)
                    
                    x_ = f.x
                    y_ = f.y
                    i1 = f.indices
                    
                    X_.append(x_)
                    Y_.append(y_)
                    indices.append(i1)
                    positions.append(pi)
                    
                X_ = np.array(X_, dtype = np.uint8)
                Y_ = np.array(Y_, dtype = np.uint8)
                indices = np.array(indices, dtype = np.int32)
                positions = np.array(positions, dtype = np.int32)
                
                comm.send([ix, X_, Y_, indices, positions, P.pop()], dest = 0)
                
    else:
        n_received = 0
        current_chunk = 0

        while n_received < len(ms_files):
            ix, x, y, indices, pi, p = comm.recv(source = MPI.ANY_SOURCE)
            logging.info('writing set {}...'.format(current_chunk))
            
            ofile.create_dataset('{}/x_0'.format(ix), data = x, compression = 'lzf')
            ofile.create_dataset('{}/y'.format(ix), data = y, compression = 'lzf')
            ofile.create_dataset('{}/indices'.format(ix), data = indices, compression = 'lzf')
            ofile.create_dataset('{}/ix'.format(ix), data = pi, compression = 'lzf')
            ofile.create_dataset('{}/pos'.format(ix), data = np.array(p), compression = 'lzf')
            
            current_chunk += 1
            n_received += 1

            ofile.flush()

        ofile.close()
    # ${code_blocks}

if __name__ == '__main__':
    main()


