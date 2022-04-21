# -*- coding: utf-8 -*-
import os
import numpy as np
import logging, argparse

import configparser

def parse_args():
    # Argument Parser    
    parser = argparse.ArgumentParser()
    # my args
    parser.add_argument("--verbose", action = "store_true", help = "display messages")
    parser.add_argument("--odir", default = "/proj/dschridelab/introgression_data/")
    parser.add_argument("--n_jobs", default = "1000")
    parser.add_argument("--n_replicates", default = "100")

    # simulation SLiM script
    parser.add_argument("--slim_file", default = "src/SLiM/introg_bidirectional.slim")

    # parameters for the simulation
    parser.add_argument("--st", default = "4")
    parser.add_argument("--mt", default = "0.25")
    parser.add_argument("--mp", default = "1")
    parser.add_argument("--phys_len", default = "3000")
    parser.add_argument("--donor_pop", default="1")

    parser.add_argument("--n_per_pop", default = "64")

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
        logging.debug("running in verbose mode")
    else:
        logging.basicConfig(level=logging.INFO)

    if not os.path.exists(args.odir):
        os.mkdir(args.odir)
        logging.debug('root: made output directory {0}'.format(args.odir))
    else:
        os.system('rm -rf {0}'.format(os.path.join(args.odir, '*')))

    # config file with the population size in it (maybe other stuff)?
    config = configparser.ConfigParser()
    config['info'] = {'n_per_pop': args.n_per_pop, 'n_per_file': str(int(args.n_replicates) // int(args.n_jobs))}

    with open(os.path.join(args.odir, 'info.config'), 'w') as configfile:
        config.write(configfile)

    return args

def main():
    args = parse_args()
    
    # scriptName, numReps, physLen, donorPop, introgLogFileName, nPerPop, splitTimeCoefficient, migrationTimeCoefficient, migrationProbability
    cmd = 'sbatch --mem=4G -t 02:00:00 -o {10} --wrap "python3 src/data/runAndParseSlim.py {0} {1} {2} {3} {4} {5} {6} {7} {8} > {9} && gzip {4} {9}"'
    
    for ix in range(int(args.n_jobs)):
        ofile_ms = os.path.join(args.odir, '{0:05d}.ms'.format(ix))
        ofile_introg = os.path.join(args.odir, '{0:05d}_introg.log'.format(ix))
        ofile_log = os.path.join(args.odir, '{0:05d}_introg.out'.format(ix))
        
        cmd_ = cmd.format(args.slim_file, args.n_replicates, args.phys_len,
                          args.donor_pop, ofile_introg, args.n_per_pop, args.st, args.mt, args.mp, ofile_ms, ofile_log)
        
        os.system(cmd_)
        
if __name__ == '__main__':
    main()

        