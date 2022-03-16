# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
import os
import numpy as np
import logging, argparse

from data_functions import load_data, get_windows

import h5py

import matplotlib.pyplot as plt
from format_unet_data_h5 import Formatter

from calc_stats_ms import *
from mpi4py import MPI

import time
import configparser
import sys

def get_feature_vector(mutation_positions, genotypes, ref_geno, arch):
    n_samples = len(genotypes[0])

    n_sites = 50000

    ## set up S* stuff -- remove mutations found in reference set
    t_ref = list(map(list, zip(*ref_geno)))
    t_geno = list(map(list, zip(*genotypes)))
    pos_to_remove = set()  # contains indexes to remove
    s_star_haps = []
    for idx, hap in enumerate(t_ref):
        for jdx, site in enumerate(hap):
            if site == 1:
                pos_to_remove.add(jdx)

    for idx, hap in enumerate(t_geno):
        s_star_haps.append([v for i, v in enumerate(hap) if i not in pos_to_remove])

    ret = []

    # individual level
    for focal_idx in range(0, n_samples):
        calc_N_ton = N_ton(genotypes, n_samples, focal_idx)
        dist = distance_vector(genotypes, focal_idx)
        min_d = [min_dist_ref(genotypes, ref_geno, focal_idx)]
        ss = [s_star_ind(np.array(s_star_haps), np.array(mutation_positions), focal_idx)]
        n_priv = [num_private(np.array(s_star_haps), focal_idx)]
        focal_arch = [row[focal_idx] for row in arch]
        lab = label(focal_arch, mutation_positions, n_sites, 0.7, 0.3)

        output = calc_N_ton + dist + min_d + ss + n_priv + lab
        ret.append(output)

    return np.array(ret, dtype = np.float32)

def pad_matrices(features, positions):
    max_window_size = max([u.shape[0] for u in features])

    features = [np.pad(u, ((0, max_window_size - u.shape[0]), (0, 0), (0, 0)), 'constant') for u in features]
    positions = [np.pad(u, ((0, max_window_size - u.shape[0]), (0, 0)), 'constant') for u in positions]

    return np.array(features, dtype = np.float32), np.array(positions, dtype = np.uint8)


def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser()
    # my args
    parser.add_argument("--verbose", action = "store_true", help = "display messages")
    parser.add_argument("--idir", default = "None")
    parser.add_argument("--chunk_size", default="4")
    parser.add_argument("--ofile", default = "archie_data.hdf5")
    parser.add_argument("--sorting", default = "seriate_match")
    
    parser.add_argument("--pop_sizes", default = "100,100")
    parser.add_argument("--out_shape", default = "2,112,128")

    parser.add_argument("--n_per_dir", default = "100")
    parser.add_argument("--pop_size", default = "100")
    parser.add_argument("--zero", action = "store_true")

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
    chunk_size = int(args.chunk_size)
        
    if comm.rank != 0:
        for ix in range(comm.rank - 1, len(idirs), comm.size - 1):
            ms = os.path.join(idirs[ix], 'mig.msOut')
            anc = os.path.join(idirs[ix], 'out.anc')

            X, Y, P = load_data(ms, anc, leave_out_last = True)
            
            # toss out the last two individuals
            for k in range(len(X)):
                X[k] = X[k][:-2,:]
                Y[k] = Y[k][:-2,:]

            f = Formatter(X, Y, sorting = args.sorting, pop = "0", 
                          pop_sizes = pop_sizes, shape = out_shape)
            
            logging.info('formatting data for idir {}...'.format(ix))
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


