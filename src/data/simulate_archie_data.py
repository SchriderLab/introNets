#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 17:21:48 2022

@author: kilgoretrout
"""

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

def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser()
    # my args
    parser.add_argument("--verbose", action = "store_true", help = "display messages")
    parser.add_argument("--n_samples", default = "1000")
    parser.add_argument("--n_jobs", default = "100")
    
    parser.add_argument("--ifile", default = "None")
    parser.add_argument("--direction", default = "ba")

    parser.add_argument("--sample", action = "store_true")
    parser.add_argument("--debug", action = "store_true")
    
    parser.add_argument("--migTime", default = "None")
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

    slurm_cmd = 'sbatch -t 1-00:00:00 --mem=8G -o {0} --wrap "{1}"'
    n = int(args.n_samples)
    
    T = 5e-4
    R = 4e-4
    A = 1
    L = 50000
    
    for ix in range(int(args.n_jobs)):
        odir = os.path.join(args.odir, '{0:04d}'.format(ix))
        os.system('mkdir -p {}'.format(odir))
        
        cmd = "cd %s; %s 202 %d -T -t %d -r %d %d -I 4 100 100 1 1 g  -en 0 1 1  -es 0.05 1 %d -ej 0.05 5 3  -ej 0.0625 2 1 -en 0.15 3 0.01 -en 0.153 3 1 -ej 0.175 4 3 -ej 0.3 3 1| tail -n1" % (odir, os.path.join(os.getcwd(), 'msmodified/ms'), n, T, R, L, A)
        print(cmd)
        
        out_file = os.path.join(args.odir, '{0:04d}.log'.format(ix))
        
        scmd = slurm_cmd.format(out_file, cmd)
        os.system(scmd)
        
if __name__ == '__main__':
    main()
        