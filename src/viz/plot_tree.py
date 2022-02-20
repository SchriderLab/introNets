# -*- coding: utf-8 -*-
import os
import argparse
import logging

import numpy as np
import matplotlib.pyplot as plt

import pickle
import networkx as nx

import pandas as pd
from networkx.drawing.nx_agraph import graphviz_layout
from networkx import optimize_graph_edit_distance
import copy

# use this format to tell the parsers
# where to insert certain parts of the script
# ${imports}

def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser()
    # my args
    parser.add_argument("--verbose", action = "store_true", help = "display messages")
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

    edges = pickle.load(open('graph_sample.pkl', 'rb'))
    
    for e in edges:
        e = list(e.T)
        
        G = nx.DiGraph()
        G.add_edges_from(e)
        
        total_seen = 1
        
        n_iterations = 0
        
        s = [-1]
        while True:
    
            new_s = []
            for node in s:
                new_s.extend(G.neighbors(node))
                
            n_iterations += 1
                
            if len(new_s) == 0:
                break
            
            total_seen += len(new_s)
            s = copy.copy(new_s)
        
        print(n_iterations)
        

    # ${code_blocks}

if __name__ == '__main__':
    main()

