# -*- coding: utf-8 -*-
import os
import numpy as np
import logging, argparse

from data_functions import load_data, get_windows

import h5py

import matplotlib.pyplot as plt

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

    parser.add_argument("--batch_size", default="8")

    parser.add_argument("--ofile", default = "archie_data.hdf5")
    parser.add_argument("--window_size", default = "50000")

    parser.add_argument("--n_per_dir", default = "100")
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
    n_per_dir = int(args.n_per_dir)
    
    n_sims = len(idirs) * n_per_dir
    window_size = int(args.window_size)
    pop_size = int(args.pop_size)
    
    if comm.rank != 0:
        for ix in range(comm.rank - 1, len(idirs), comm.size - 1):
            ms = os.path.join(idirs[ix], 'mig.msOut')
            anc = os.path.join(idirs[ix], 'out.ADMIXED.anc')

            X, Y, P = load_data(ms, anc, leave_out_last = True)

            for k in range(len(X)):
                #logging.info('{0}: working on file {1}, dataset {2}'.format(comm.rank, ix, k))

                x = X[k]
                ipos = P[k]
                y = Y[k]
                
                ipos = np.round(np.array(ipos) * window_size).astype(np.int32)

                features = []

                genotypes = list(map(list, list(x[:pop_size,:].T.astype(np.uint8))))
                ref_geno = list(map(list, list(x[pop_size:pop_size * 2,:].T.astype(np.uint8))))
                mutation_positions = list(ipos.astype(np.int32))
                arch = list(map(list, list(y.T.astype(np.uint8).astype(str))))

                f = get_feature_vector(mutation_positions, genotypes, ref_geno, arch)

                comm.send([f], dest = 0)

    else:
        ofile = h5py.File(args.ofile, 'w')

        n_recieved = 0
        batch_size = int(args.batch_size)

        counter = 0

        features = []

        while n_recieved < n_sims:
            f = comm.recv(source = MPI.ANY_SOURCE)

            n_recieved += 1

            if n_recieved % 10 == 0:
                logging.info('0: recieved {0} simulations'.format(n_recieved))

            features.append(f)

            while len(features) >= batch_size:
                ofile.create_dataset('{0}/features'.format(counter), data = np.array(features[-batch_size:], dtype = np.float32), compression = 'gzip')
                ofile.flush()

                del features[-batch_size:]

                counter += 1
                
                logging.info('0: wrote batch {} ...'.format(counter))

        logging.debug('0: closing file...')
        ofile.close()


if __name__ == '__main__':
    main()

# sbatch -p 528_queue -n 512 -t 1-00:00:00 --wrap "mpirun -oversubscribe python3 src/data/format_data_ghost.py --idir /proj/dschridelab/introgression_data/sims_64_10e5_ghost/ --ofile /proj/dschridelab/ddray/archie_64_data.hdf5 --verbose"
# sbatch -p 528_queue -n 512 -t 1-00:00:00 --wrap "mpirun -oversubscribe python3 src/data/format_data_ghost.py --idir /proj/dschridelab/introgression_data/sims_200_10e5_ghost/ --ofile /proj/dschridelab/ddray/archie_200_data.hdf5 --verbose --n_individuals 200 --window_size 50000"
