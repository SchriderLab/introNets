# -*- coding: utf-8 -*-
import os
import argparse
import logging

from data_functions import seriate_x, read_ms_tree
from scipy.spatial.distance import pdist, squareform
import glob

import networkx as nx
import itertools
import numpy as np
import random
import matplotlib.pyplot as plt

# return the first and last index of the chunks of size n of lst
def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    ret = []
    
    for i in range(0, len(lst), n):
        l = lst[i:i + n]
        
        ret.append((l[0], l[-1]))
        
    return ret


# use this format to tell the parsers
# where to insert certain parts of the script
# ${imports}

def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser()
    # my args
    parser.add_argument("--verbose", action = "store_true", help = "display messages")
    parser.add_argument("--idir", default = "None")
    parser.add_argument("--N", default = "128")
    parser.add_argument("--pop_sizes", default = "20,14")
    parser.add_argument("--archie", action = "store_true")
    parser.add_argument("--L", default = "10000")

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
    
    pop_sizes = tuple(list(map(int, args.pop_sizes.split(','))))
    pop0 = list(range(pop_sizes[0]))
    pop1 = list(range(pop_sizes[0], pop_sizes[0] + pop_sizes[1]))
    
    sample_size = sum(pop_sizes)
    # size of the windows to perform seriation on
    N = int(args.N)
    
    # list of the integer labels of the sample individuals (just the first N)
    current_day_nodes = list(range(sample_size))
    if args.archie:
        current_day_nodes = current_day_nodes[:-2]
    
    indices = list(itertools.combinations(current_day_nodes, 2))
    
    idirs = sorted(glob.glob(os.path.join(args.idir, '*')))
    random.shuffle(idirs)
    
    metrics = ['cosine', 'dice', 'cityblock', 'euclidean', 'russellrao']
    
    bl = []
    hops = []
    
    # make a dictionary with lists of each metric
    metric = dict()
    for m in metrics:
        metric[m] = []
    
    for idir in idirs[:1]:
        print('working on {}...'.format(idir))
        
        ifile = os.path.join(idir, 'mig.msOut.gz')
        if not os.path.exists(ifile):
            continue
        
        T = read_ms_tree(ifile, n = sample_size, L = int(args.L))
        
        # list of lists of graphs
        E = T['edges']
        
        # list of lists of positions for the graph sequence (start and end tuple)
        pos = T['positions']
        
        # alignment matrices
        als = T['alignment']
        
        print('have {} replicates...'.format(len(E)))
        
        for ii in range(len(E)):
            E_ = E[ii]
            pos_ = pos[ii]
            
            x = np.array(als[ii], dtype = np.float32).T
            if args.archie:
                x = x[:-2,:]
            
            ix = list(range(x.shape[1]))
            
            # indices to store trees that actually span any
            # introgressed sites
            ix_ = []
            
            print('getting distances for replicate {}...'.format(ii + 1))
            Ds = []
            for ij in range(len(E_)):
                edges = E_[ij]
                p0, p1 = pos_[ij]
                if p1 - p0 == 0:
                    continue
                
                G = nx.Graph()
                for k in range(len(edges)):
                    G.add_edge(edges[k][0], edges[k][1], branch_length = edges[k][2], hop = 0.5)
            
                paths = nx.shortest_path(G)

                D = np.array([len(paths[i][j]) for (i,j) in indices]) / 2.
                
                D_branch = []
                for i,j in indices:
                    path = paths[i][j]
                    
                    _ = [G.edges[path[k], path[k + 1]]['branch_length'] for k in range(len(path) - 1)]
                    for ii in range(len(_)):
                        if _[ii] is None:
                            _[ii] = 0.
    
                    D_branch.append(sum(_))
                    
                Ds.append((D, D_branch))
                ix_.append(ij)
                
            c = chunks(ix, 128)
            
            for ij in range(len(c)):
                ix = []
                weights = []
                
                i, j = c[ij]
                if j - i != N - 1:
                    continue
                
                for ik in range(len(ix_)):
                    p0, p1 = pos_[ix_[ik]]
                    
                    if (p0 < i and p1 < j) or (p0 > i and p1 > j):
                        continue
                    
                    if p0 >= i and p1 <= j:
                        l = p1 - p0
                        
                    elif p0 >= i:
                        l = j - p0
                    else:
                        l = p1 - i
                        
                    w = l / N
                    
                    weights.append(w)
                    ix.append(ik)
       
                weights = np.array(weights).reshape(len(weights), 1)
                
                D = np.array([Ds[u][0] for u in ix])
                D_branch = np.array([Ds[u][1] for u in ix])
                
                D = np.sum(D * weights, axis = 0)
                D_branch = np.sum(D * weights, axis = 0)
                
                for m in metrics:
                    D_ = pdist(x[:,i:j + 1], metric = m)
                    
                    metric[m].extend(D_)
                    
                bl.extend(D)
                hops.extend(D_branch)
                
    for m in metrics:
        D = metric[m]

        print('distance metric: {}'.format(m))
        print('correlation <-> shortest branch length distance: {}'.format(np.corrcoef(D, bl)[0, 1]))
        print('correlation <-> shortest n branches distance: {}'.format(np.corrcoef(D, hops)[0, 1]))
        
        plt.scatter(D, bl, color = 'k', alpha = 0.5)
        plt.show()
                        
                
    

    # ${code_blocks}

if __name__ == '__main__':
    main()

