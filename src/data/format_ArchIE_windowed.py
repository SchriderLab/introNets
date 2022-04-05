# -*- coding: utf-8 -*-
import os
import numpy as np
import logging, argparse

from data_functions import load_data, get_windows

import h5py

import matplotlib.pyplot as plt

from data_functions import get_feature_vector
from mpi4py import MPI

import time
import configparser
import sys

def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser()
    # my args
    parser.add_argument("--verbose", action = "store_true", help = "display messages")
    parser.add_argument("--idir", default = "None")

    parser.add_argument("--batch_size", default="8")

    parser.add_argument("--ofile", default = "archie_data.hdf5")
    parser.add_argument("--window_size", default = "50000")
    parser.add_argument("--step", default = "10000")
    parser.add_argument("--n", default = "10e6")

    parser.add_argument("--n_per_dir", default = "10")
    parser.add_argument("--pop_size", default = "100")

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
        logging.debug("running in verbose mode")
    else:
        logging.basicConfig(level=logging.INFO)

    return args

def main():
    # configure MPI
    comm = MPI.COMM_WORLD

    args = parse_args()

    idirs = [os.path.join(args.idir, u) for u in os.listdir(args.idir) if not '.' in u]
    idirs_ = []
    for idir in idirs:
        ms = os.path.join(idir, 'mig.msOut')
        anc = os.path.join(idir, 'out.ADMIXED.anc')
        
        if os.path.exists(ms) and os.path.exists(anc):
            idirs_.append(idir)
    
    idirs = idirs_
    
    print(idirs)
    
    n_per_dir = int(args.n_per_dir)
    
    n_sims = len(idirs) * n_per_dir
    pop_size = int(args.pop_size)
    
    # the number of segsites in each simulation
    N = int(float(args.n))
    # window size to compute stats on
    window_size = int(args.window_size)
    # step size for sliding windows
    step_size = int(args.step)
    
    if comm.rank != 0:
        for ix in range(comm.rank - 1, len(idirs), comm.size - 1):
            logging.info('working on {}...'.format(idirs[ix]))
            
            ms = os.path.join(idirs[ix], 'mig.msOut')
            anc = os.path.join(idirs[ix], 'out.ADMIXED.anc')

            X, Y, P = load_data(ms, anc, leave_out_last = True)
            
            for k in range(len(X)):
                logging.info('{0}: working on file {1}, dataset {2}'.format(comm.rank, ix, k))
                
                F = []
                pos = []
                
                x = X[k]
                y = Y[k]
                
                p = np.round(np.array(P[k]) * N).astype(np.int32)
                
                indices = get_windows(p, window_size, step_size)

                logging.info('have {} windows...'.format(len(indices)))                
                for ii in indices:
                    genotypes = list(map(list, list(x[:pop_size,ii].T.astype(np.uint8))))
                    ref_geno = list(map(list, list(x[pop_size:pop_size * 2,ii].T.astype(np.uint8))))
                    mutation_positions = list(p[ii].astype(np.int32))
                    arch = list(map(list, list(y[:,ii].T.astype(np.uint8).astype(str))))

                    f = get_feature_vector(mutation_positions, genotypes, ref_geno, arch)
                
                    pos.append([ii[0], ii[-1]])
                    F.append(f)
                    
                comm.send([F, pos, y], dest = 0)
                
    else:
        ofile = h5py.File(args.ofile, 'w')

        n_recieved = 0
        
        while n_recieved < n_sims:
            F, pos, y = comm.recv(source = MPI.ANY_SOURCE)
                    
            ofile.create_dataset('{0}/f'.format(n_recieved), data = np.array(F, dtype = np.float32), compression = 'gzip')
            ofile.create_dataset('{0}/pos'.format(n_recieved), data = np.array(pos, dtype = np.int32), compression = 'gzip')
            ofile.create_dataset('{0}/y'.format(n_recieved), data = y, compression = 'gzip')
            
            logging.info('wrote dataset {}...'.format(n_recieved))
            n_recieved += 1
            
            ofile.flush()
            
        ofile.close()
        
if __name__ == '__main__':
    main()
            
            
            

            