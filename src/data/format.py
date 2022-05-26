# -*- coding: utf-8 -*-
import os
import argparse
import logging

import glob
import numpy as np

from data_functions import load_data, TwoPopAlignmentFormatter, load_data_slim, read_slim_out

from mpi4py import MPI
import h5py

# use this format to tell the parsers
# where to insert certain parts of the script
# ${imports}

import time
            
def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser()
    # my args
    parser.add_argument("--verbose", action = "store_true", help = "display messages")
    parser.add_argument("--idir", default = "None")
    parser.add_argument("--chunk_size", default = "4")

    parser.add_argument("--ofile", default = "None")
    parser.add_argument("--sorting", default = "seriate_match")
    parser.add_argument("--metric", default = "cosine")
    
    parser.add_argument("--pop_sizes", default = "64,64")
    parser.add_argument("--out_shape", default = "2,128,128")
    
    parser.add_argument("--densify", action = "store_true", help = "remove singletons")
    parser.add_argument("--include_zeros", action = "store_true")
    
    parser.add_argument("--pop", default = "0", help = "only return y values for one pop?")

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
    if all([('.' in u) for u in idirs]):
        ms_files = sorted(glob.glob(os.path.join(args.idir, '*.ms.gz')))
        anc_files = sorted(glob.glob(os.path.join(args.idir, '*.log.gz')))
        log_files = sorted(glob.glob(os.path.join(args.idir, '*.out')))
        
        idirs = list(zip(ms_files, anc_files, log_files))
        slim = True
    else:
        slim = False
        
    if comm.rank == 0:
        logging.info('found {} directories to read...'.format(len(idirs)))
        
    chunk_size = int(args.chunk_size)
    
    pop_sizes = tuple(list(map(int, args.pop_sizes.split(','))))
    out_shape = tuple(list(map(int, args.out_shape.split(','))))
    
    n_ind = sum(pop_sizes)

    if comm.rank != 0:
        for ix in range(comm.rank - 1, len(idirs), comm.size - 1):
            logging.info('{0}: on {1}...'.format(comm.rank, ix))
            
            # are we formatting SLiM or MS data?  
            # auto-detected above
            # time the disk-read time
            t0 = time.time()
            if not slim:
                idir = idirs[ix]
                
                msFile = os.path.join(idir, 'mig.msOut.gz')
                ancFile = os.path.join(idir, 'out.anc.gz')
                
                if not (os.path.exists(msFile) and os.path.exists(ancFile)):
                    logging.info('{0}: have no data for {1}...'.format(comm.rank, idir))    
                    continue
                
                x, y, _ = load_data(msFile, ancFile, n = n_ind)
                
                if os.path.exists(os.path.join(idir, 'mig.tbs')):
                    params = list(np.loadtxt(os.path.join(idir, 'mig.tbs')))
                else:
                    params = None
            else:
                msFile, ancFile, out = idirs[ix]
                
                mp, mt = read_slim_out(out)
                mp = np.array(mp).reshape(-1, 2)
                mt = np.array(mt).reshape(-1, 1)
                
                params = np.hstack([mp, mt])
                x, _, y = load_data_slim(msFile, ancFile, n_ind)
            
            t_disk = time.time() - t0
            
            if len(y) == 0:
                y = None
                
            f = TwoPopAlignmentFormatter(x, y, params, sorting = args.sorting, sorting_metric = args.metric, pop = int(args.pop), 
                          pop_sizes = pop_sizes, shape = out_shape)
            f.format(include_zeros = args.include_zeros)
            logging.debug('{3}: took an average {0} s to seriate, {1} to match and {2} to read the data...'.format(np.mean(f.time[0]), 
                                                                                                                   np.mean(f.time[1]), t_disk, comm.rank))
        
            comm.send([f.x, f.y, f.p], dest = 0)
    else:
        n_received = 0
        current_chunk = 0
        
        no_y = False

        X = []
        Y = []
        params = []
        while n_received < len(idirs):
            x, y, p = comm.recv(source = MPI.ANY_SOURCE)
            
            if y is None:
                no_y = True
            else:
                Y.extend(y)
            
            X.extend(x)
            Y.extend(y)
            params.extend(p)
            
            n_received += 1
            
            while len(X) > chunk_size:
                if not no_y:
                    ofile.create_dataset('{0}/y'.format(current_chunk), data = np.array(Y[-chunk_size:], dtype = np.uint8), compression = 'lzf')
                    del Y[-chunk_size:]
                    
                if len(params) > 0:
                    ofile.create_dataset('{0}/params'.format(current_chunk), data = np.array(params[-chunk_size:], dtype = np.float32), compression = 'lzf')
                    del params[-chunk_size:]
                
                ofile.create_dataset('{0}/x_0'.format(current_chunk), data = np.array(X[-chunk_size:], dtype = np.uint8), compression = 'lzf')
                ofile.flush()
                
                del X[-chunk_size:]

                logging.info('0: wrote chunk {0}'.format(current_chunk))
                
                current_chunk += 1
                
        ofile.close()

if __name__ == '__main__':
    main()
    
# mpirun python3 src/data/format.py --idir /pine/scr/d/d/ddray/dros_sims/AB --out_shape 2,32,128 --pop_sizes 20,14 --ofile /pine/scr/d/d/ddray/dros_sims/AB_n128.hdf5 pop 1
