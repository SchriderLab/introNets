# -*- coding: utf-8 -*-
import os, sys

import numpy as np
import argparse
import logging

from data_functions import writeTbsFile
import copy

# this function creates an array for writing to text that has the ms parameters
# from a CSV file produced via bootstrapped DADI runs
# the values expected in the CSV our given below
def parameters_df(df, ix, thetaOverRho, migTime, migProb, n):
    # estimated from ??
    u = 5.0e-9
    L = 10000
    
    # Isolation, migration model
    # estimated with DADI (default)
    # nu1 and nu2 (before)
    # nu1_0 and nu2_0 (after split)
    # migration rates (Nref_m12, Nref_m21)
    ll, aic, Nref, nu1_0, nu2_0, nu1, nu2, T, Nref_m12, Nref_m21 = df[ix]
    
    nu1_0 /= Nref
    nu2_0 /= Nref
    nu1 /= Nref
    nu2 /= Nref
    
    T /= (4*Nref / 15.)
    
    alpha1 = np.log(nu1/nu1_0)/T
    alpha2 = np.log(nu2/nu2_0)/T
    
    theta = 4 * Nref * u * L
    rho = theta / thetaOverRho
    
    migTime = migTime * T
    
    p = np.tile(np.array([theta, rho, nu1, nu2, alpha1, alpha2, 0, 0, T, T, migTime, 1 - migProb, migTime]), (n, 1)).astype(object)
    
    return p, ll, Nref

def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser()
    # my args
    parser.add_argument("--verbose", action = "store_true", help = "display messages")
    parser.add_argument("--n_samples", default = "1000", help = "number of alignments to simulate per job")
    parser.add_argument("--n_jobs", default = "1", help = "number of jobs.  If your on a system without SLURM, then this should be left to 1 (default)")
    
    parser.add_argument("--ifile", default = "None", help = "CSV file of bootstrapped demographic estimates. only applicable for the drosophila case; --model dros")
    
    parser.add_argument("--model", default = "dros", help = "model you'd like to simulate. current options our 'archie' and 'dros'")
    parser.add_argument("--direction", default = "ab", help = "directionality of migration. only applicable for the drosophila case; --model dros")
    parser.add_argument("--slurm", action = "store_true")
    
    parser.add_argument("--window_size", default = "50000", help = "size in base pairs of the region to simulate")
    
    parser.add_argument("--mt_range", default = "None")
    parser.add_argument("--t_range", default = "None")

    parser.add_argument("--odir", default = "None", help = "directory for the ms output and logs to be written to")
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
    
    # the supported models for this script
    if not (args.model in ['dros', 'archie']):
        print('not a supported model for this script...')
        print("valid options for --model: 'archie', 'dros'")
        return
    
    if args.model == 'dros':
        df = np.loadtxt(args.ifile)
    
    slurm_cmd = 'sbatch -t 1-00:00:00 --mem=8G -o {0} --wrap "{1}"'
    n = int(args.n_samples)
    L = int(args.window_size)
    
    counter = 0
    
    for j in range(int(args.n_jobs)):
        if args.model == 'dros':
            for ix in range(df.shape[0]):
                # size of the populations
                SIZE_A = 20
                SIZE_B = 14
                
                L = 10000
                
                P, ll, Nref = parameters_df(df, ix, 1., 0., 0., n)
                if ll > -2000:
                    odir = os.path.join(args.odir, 'iter{0:06d}'.format(counter))
                    counter += 1
                    
                    os.system('mkdir -p {}'.format(odir))
                    
                    if args.mt_range != "None":
                        mt_range = tuple(list(map(float, args.mt_range.split(','))))
                        
                        # replace mean migTime and the rest with a uniformly random distribution around it
                        migTime = np.random.uniform(mt_range[0], mt_range[1], (P.shape[0], ))
                    
                    
                    migProb = 1 - np.random.uniform(0., 1.0, (P.shape[0], ))
                    rho = np.random.uniform(0.1, 0.3, (P.shape[0], ))
                    
                    if args.t_range != "None":
                        t_range = tuple(list(map(float, args.t_range.split(','))))
                        
                        T = copy.copy(P[:,-4])
                        
                        # rescale alpha1 and alpha2
                        P[:,4] *= T
                        P[:,5] *= T
                        
                        P[:,-4] = np.random.uniform(t_range[0], t_range[1], (P.shape[0], ))
                        P[:,-4] /= (4*Nref / 15.)
                        
                        # rescale alpha1 and alpha2
                        P[:,4] /= P[:,-4]
                        P[:,5] /= P[:,-4]
                    
                    P[:,-1] = migTime * P[:,-4]
                    P[:,-3] = migTime * P[:,-4]
                    
                    P[:, 1] = P[:,0] / rho
                    P[:,-2] = migProb
                
                    if ('ba' in args.direction) or ('ab' in args.direction):
                        writeTbsFile(np.concatenate([P, np.random.randint(0, 2**14, size = (P.shape[0], 3))], axis = 1), os.path.join(odir, 'mig.tbs'))
                    else:
                        writeTbsFile(np.concatenate([P[:,:-3], np.random.randint(0, 2**14, size = (P.shape[0], 1))], axis = 1), os.path.join(odir, 'mig.tbs'))
                        
                    # direction + wtrees (if you desire the output as a series of genealogical trees)
                    # in the case of genealogical trees, we use regular ms as we observed an error in msmodified when trying...
                    if args.direction == 'ab':
                        cmd = "cd %s; %s %d %d -t tbs -r tbs %d -I 2 %d %d -n 1 tbs -n 2 tbs -eg 0 1 tbs -eg 0 2 tbs -ma x tbs tbs x -ej tbs 2 1 -en tbs 1 1 -es tbs 2 tbs -ej tbs 3 1 -seeds tbs tbs tbs < %s | tee %s" % (odir, os.path.join(os.getcwd(), 'msmodified/ms'), SIZE_A + SIZE_B, len(P), L, SIZE_A, SIZE_B, 'mig.tbs', 'mig.msOut')
                    elif args.direction == 'ba':
                        cmd = "cd %s; %s %d %d -t tbs -r tbs %d -I 2 %d %d -n 1 tbs -n 2 tbs -eg 0 1 tbs -eg 0 2 tbs -ma x tbs tbs x -ej tbs 2 1 -en tbs 1 1 -es tbs 1 tbs -ej tbs 3 2 -seeds tbs tbs tbs < %s | tee %s" % (odir, os.path.join(os.getcwd(), 'msmodified/ms'), SIZE_A + SIZE_B, len(P), L, SIZE_A, SIZE_B, 'mig.tbs', 'mig.msOut')
                    elif args.direction == 'none':
                        cmd = "cd %s; %s %d %d -t tbs -r tbs %d -I 2 %d %d -n 1 tbs -n 2 tbs -eg 0 1 tbs -eg 0 2 tbs -ma x tbs tbs x -ej tbs 2 1 -en tbs 1 1 -seed tbs < %s | tee %s" % (odir, os.path.join(os.getcwd(), 'msdir/ms'), SIZE_A + SIZE_B, len(P), L, SIZE_A, SIZE_B, 'mig.tbs', 'mig.msOut')
                    elif args.direction == 'ab_wtrees':
                        cmd = "cd %s; %s %d %d -t tbs -r tbs %d -I 2 %d %d -n 1 tbs -n 2 tbs -eg 0 1 tbs -eg 0 2 tbs -ma x tbs tbs x -ej tbs 2 1 -en tbs 1 1 -es tbs 2 tbs -ej tbs 3 1 -seeds tbs tbs tbs -T < %s | tee %s" % (odir, os.path.join(os.getcwd(), 'msdir/ms'), SIZE_A + SIZE_B, len(P), L, SIZE_A, SIZE_B, 'mig.tbs', 'mig.msOut')
                    elif args.direction == 'ba_wtrees':
                        cmd = "cd %s; %s %d %d -t tbs -r tbs %d -I 2 %d %d -n 1 tbs -n 2 tbs -eg 0 1 tbs -eg 0 2 tbs -ma x tbs tbs x -ej tbs 2 1 -en tbs 1 1 -es tbs 1 tbs -ej tbs 3 2 -seeds tbs tbs tbs -T < %s | tee %s" % (odir, os.path.join(os.getcwd(), 'msdir/ms'), SIZE_A + SIZE_B, len(P), L, SIZE_A, SIZE_B, 'mig.tbs', 'mig.msOut')
                    elif args.direction == 'none_wtrees':
                        cmd = "cd %s; %s %d %d -t tbs -r tbs %d -I 2 %d %d -n 1 tbs -n 2 tbs -eg 0 1 tbs -eg 0 2 tbs -ma x tbs tbs x -ej tbs 2 1 -en tbs 1 1 -seed tbs -T < %s | tee %s" % (odir, os.path.join(os.getcwd(), 'msdir/ms'), SIZE_A + SIZE_B, len(P), L, SIZE_A, SIZE_B, 'mig.tbs', 'mig.msOut')
                    
                    # print the command, do the command, gzip the outputs
                    cmd = "echo '{0}' && {0} && gzip mig.msOut out.anc".format(cmd)
                    print('simulating via the recommended parameters...')
                    sys.stdout.flush()
                    
                    if args.slurm:
                        fout = os.path.join(odir, 'slurm.out')
                        cmd = slurm_cmd.format(fout, cmd)
                        
                    os.system(cmd)
        
        elif args.model == 'archie':
            T = 5e-4 * L
            R = 4e-4 * L
            A = 1 - 0.02
            
            seeds = np.random.randint(0, 2**14, size = (3,))
            
            cmd = "cd {0}; {1} 202 {2} -t {3} -r {4} {5} -I 4 100 100 1 1 g  -en 0 1 1  -es 0.05 1 {6} -ej 0.05 5 3  -ej 0.0625 2 1 -en 0.15 3 0.01 -en 0.153 3 1 -ej 0.175 4 3 -ej 0.3 3 1 -seeds {7} {8} {9} | tee mig.msOut".format(odir, os.path.join(os.getcwd(), 'msmodified/ms'), n, T, R, L, A, seeds[0], seeds[1], seeds[2])
        
            cmd = "echo '{0}' && {0} && gzip mig.msOut out.anc".format(cmd)
            print('simulating via the recommended parameters...')
            sys.stdout.flush()
            
            if args.slurm:
                fout = os.path.join(odir, 'slurm.out')
                cmd = slurm_cmd.format(fout, cmd)
                
            os.system(cmd)
            
if __name__ == '__main__':
    main()

