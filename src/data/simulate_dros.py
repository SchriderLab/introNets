# -*- coding: utf-8 -*-
import os, random, sys
# import runCmdAsJob

import numpy as np
import os
import argparse
import logging
import subprocess
import itertools

# use this format to tell the parsers
# where to insert certain parts of the script
# ${imports}

SIZE_A = 20
SIZE_B = 14
N_SITES = 10000

bounds = dict()
bounds[0] = (1., 150.) # theta
bounds[1] = (0.01, 0.25) # theta_rho
bounds[2] = (5., 45.) # nu_ab
bounds[3] = (0.01, 3.0) # nu_ba
bounds[4] = (0.01, None) # migTime
bounds[5] = (0.01, 0.7) # migProb
bounds[6] = (0.3, 0.8) # T
bounds[7] = (15000., 80.)

params = dict()
params['theta'] = 68.29691232
params['theta_rho'] = 0.2
params['nu_a'] = 19.022761
params['nu_b'] = 0.054715
params['m_ab'] = 0.025751
params['m_ba'] = 0.172334
params['T'] = 0.664194

maxRandMs = 2**32-1
maxRandMsMod = 2**15-1

def parameters(n, sample = False):
    p = np.random.uniform(0., 1., size = (n, 7))
    
    theta = (bounds[0][1] - bounds[0][0]) * p[:,0] + bounds[0][0]
    theta_rho = (bounds[1][1] - bounds[1][0]) * p[:,1] + bounds[1][0]
    rho = theta / theta_rho
    nu_a = (bounds[2][1] - bounds[2][0]) * p[:,2] + bounds[2][0]
    nu_b = (bounds[3][1] - bounds[3][0]) * p[:,3] + bounds[3][0]
    T = (bounds[6][1] - bounds[6][0]) * p[:,6] + bounds[6][0]

    b_ = list(bounds[4])
    b_[1] = T / 4.
    migTime = (b_[1] - b_[0]) * p[:,4] + b_[0]
    migProb = (bounds[5][1] - bounds[5][0]) * p[:,5] + bounds[5][0]
    
    p = np.vstack([theta, rho, nu_a, nu_b, np.zeros(n, dtype = np.uint8), np.zeros(n, dtype = np.uint8), T, T, migTime, 1 - migProb, migTime]).astype(object).T

    return p

def parameters_df(df, ix, thetaOverRho, migTime, migProb, n):
    u = 3.5e-9
    L = 10000
    
    ll, aic, Nref, nu1_0, nu2_0, nu1, nu2, T, Nref_m12, Nref_m21 = df[ix]
    
    nu1_0 /= Nref
    nu2_0 /= Nref
    nu1 /= Nref
    nu2 /= Nref
    
    T /= (4*Nref)
    
    alpha1 = np.log(nu1/nu1_0)/T
    alpha2 = np.log(nu2/nu2_0)/T
    
    theta = 4 * Nref * u * L
    rho = theta / thetaOverRho
    
    migTime = migTime * T
    
    p = np.tile(np.array([theta, rho, nu1, nu2, alpha1, alpha2, 0, 0, T, T, migTime, 1 - migProb, migTime]), (n, 1)).astype(object)
    
    return p

def writeTbsFile(params, outFileName):
    with open(outFileName, "w") as outFile:
        for paramVec in params:
            outFile.write(" ".join([str(x) for x in paramVec]) + "\n")

def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser()
    # my args
    parser.add_argument("--verbose", action = "store_true", help = "display messages")
    parser.add_argument("--n_samples", default = "1000")
    parser.add_argument("--n_jobs", default = "1")
    
    parser.add_argument("--ifile", default = "None")

    parser.add_argument("--sample", action = "store_true")
    # ${args}

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
    
    df = np.loadtxt(args.ifile)

    slurm_cmd = 'sbatch -t 1-00:00:00 --mem=8G -o {0} --wrap "{1}"'
    n = int(args.n_samples)
    
    rho = [0.2, 0.15]
    migTime = [0.7, 0.5, 0.3]
    migProb = [0.25, 0.5, 0.7, 0.1]
    
    p = list(itertools.product(rho, migTime, migProb))
    counter = 0
    
    for ix in range(df.shape[0]):
        for p_ in p:

            rho, migTime, migProb = p_
            
            P = parameters_df(df, ix, rho, migTime, migProb, n)
            
            odir = os.path.join(args.odir, 'iter{0:06d}'.format(counter))
            counter += 1
            
            os.system('mkdir -p {}'.format(odir))
        
            writeTbsFile(P, os.path.join(odir, 'mig.tbs'))
        
            cmd = "cd %s; %s %d %d -t tbs -r tbs %d -I 2 %d %d -n 1 tbs -n 2 tbs -eg 0 1 tbs -eg 0 2 tbs -ma x tbs tbs x -ej tbs 2 1 -en tbs 1 1 -es tbs 2 tbs -ej tbs 3 1 < %s" % (odir, os.path.join(os.getcwd(), 'msmodified/ms'), SIZE_A + SIZE_B, len(P), N_SITES, SIZE_A, SIZE_B, 'mig.tbs')
            print('simulating via the recommended parameters...')
            sys.stdout.flush()
        
            fout = os.path.join(odir, 'mig.msOut')
            os.system(slurm_cmd.format(fout, cmd))
            

if __name__ == '__main__':
    main()

