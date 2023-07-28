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
    parser.add_argument("--odir", default = "None", help = "output directory")
    parser.add_argument("--n_jobs", default = "1000", help = "number of SLiM commands to run")
    parser.add_argument("--n_replicates", default = "100", help = "number of simulation replicates to run per SLiM command")

    # simulation SLiM script
    parser.add_argument("--slim_file", default = "src/SLiM/introg_bidirectional.slim", 
                        help = "location of .slim file (instructions for the demography and other parameters, etc.")

    # parameters for the simulation
    parser.add_argument("--st", default = "4", help = "split time coefficient (see .slim file)")
    parser.add_argument("--mt", default = "0.25", help = "migration time coefficient (see .slim file)")
    parser.add_argument("--mp", default = "1", help = "migration probability (legacy isn't actually used...)")
    parser.add_argument("--sel_co", default = 0.0)
    
    parser.add_argument("--phys_len", default = "10000", help = "length of simulated chromosome in base pairs")
    parser.add_argument("--direction", default="ab", help = "directionality of migration.")
    parser.add_argument("--slurm", action="store_true", help = "whether to run locally (no SLURM)")

    parser.add_argument("--n_per_pop", default = "64", help = "number of sampled individuals in each population")

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
        logging.debug("running in verbose mode")
    else:
        logging.basicConfig(level=logging.INFO)

    if not os.path.exists(args.odir):
        os.system('mkdir -p {}'.format(args.odir))
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
    
    if args.slurm:
        # SLURM job submission
        # scriptName, numReps, physLen, donorPop, introgLogFileName, nPerPop, splitTimeCoefficient, migrationTimeCoefficient, migrationProbability
        cmd = 'sbatch --mem=4G -t 02:00:00 -o {10} --wrap "python3 src/data/runAndParseSlim.py {0} {1} {2} {3} {4} {5} {6} {7} {8} {11} | tee {9} && gzip {4} {9}"'
    else:
        cmd = 'python3 src/data/runAndParseSlim.py {0} {1} {2} {3} {4} {5} {6} {7} {8} {10} | tee {9} && gzip {4} {9}'

    # for compatibiltiy with notation in SLiM script
    # we assume if a custom script is used it is a two-population demography with such options (or it ignores them)
    # if you need custom args you'll have to modify the above command etc.
    if args.direction == "ab":
        donor_pop = "1"
    elif args.direction == "ba":
        donor_pop = "2"
    elif args.direction == "bi":
        donor_pop = "3"
    else:
        donor_pop = "0"
        
    for ix in range(int(args.n_jobs)):
        ofile_ms = os.path.join(args.odir, '{0:05d}.ms'.format(ix))
        ofile_introg = os.path.join(args.odir, '{0:05d}_introg.log'.format(ix))
        ofile_log = os.path.join(args.odir, '{0:05d}_introg.out'.format(ix))
        
        if args.slurm:
            cmd_ = cmd.format(args.slim_file, args.n_replicates, args.phys_len,
                              donor_pop, ofile_introg, args.n_per_pop, args.st, args.mt, args.mp, ofile_ms, ofile_log, args.sel_co)
            
            # submit via SLURM
            os.system(cmd_)
        else:
            cmd_ = cmd.format(args.slim_file, args.n_replicates, args.phys_len,
                              donor_pop, ofile_introg, args.n_per_pop, args.st, args.mt, args.mp, ofile_ms, args.sel_co)
        
            # save the paramaters for later (done automatically via slurm) (they are written to the standard error of the ran script)
            cmd_ = '{0}'.format(cmd_)
            print(cmd_)
            f = open(ofile_log, 'w')
        
            from subprocess import PIPE, Popen

            p = Popen(cmd_, shell=True, stdout=PIPE, stderr=PIPE)
            stdout, stderr = p.communicate()
            
            if stderr is not None:
                f.write(stderr.decode('utf-8'))
                
            f.close()
        
if __name__ == '__main__':
    main()

        