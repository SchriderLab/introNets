# -*- coding: utf-8 -*-
import os
import argparse
import logging

import networkx as nx

import sys
import copy
import h5py
import numpy as np

from mpi4py import MPI

import matplotlib
matplotlib.use('Agg')
from scipy.spatial.distance import squareform

import time
import itertools
# use this format to tell the parsers
# where to insert certain parts of the script
# ${imports}

def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser()
    # my args
    parser.add_argument("--verbose", action = "store_true", help = "display messages")
    parser.add_argument("--idir", default = "None")
    parser.add_argument("--idir_relate", default = "None")

    parser.add_argument("--ofile", default = "None")
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
    
    dnsps = []
    
    counter = 0

    idir = args.idir
    idir_relate = args.idir_relate
    
    if comm.rank == 0:
        print('working on {}...'.format(idir))
    
    anc_files = [os.path.join(idir_relate, u) for u in os.listdir(idir_relate) if u.split('.')[-1] == 'anc']
    
    comm.Barrier()
    
    if comm.rank != 0:
        for ix in range(comm.rank - 1, len(anc_files), comm.size - 1):
            ifile = os.path.join(idir, '{}.npz'.format(anc_files[ix].split('/')[-1].split('.')[0]))
            ifile = np.load(ifile)
            
            x = ifile['x']
            y = ifile['y']
            pos = ifile['positions']
            
            comm.send([ix, x, y, pos], dest = 0)
            
            snps = []
            anc_file = open(anc_files[ix], 'r')
            
            lines = anc_file.readlines()[2:]
            
            for ij in range(len(lines)):
                line = lines[ij]
                
                nodes = []
                parents = []
                lengths = []
                n_mutations = []
                regions = []
                
                edges = []
                
                # new tree
                line = line.replace(':', ' ').replace('(', '').replace(')', '').replace('\n', '')
                line = line.split(' ')[:-1]
    
                start_snp = int(line[0])
                snps.append(start_snp)
                
                try:
                    for j in range(2, len(line), 5):
                        nodes.append((j - 1) // 5)
                        
                        parents.append(int(line[j]))
                        lengths.append(float(line[j + 1]))
                        n_mutations.append(float(line[j + 2]))
                        regions.append((int(line[j + 3]), int(line[j + 4])))
                        
                        edges.append((parents[-1], nodes[-1]))
                except:
                    break
                
                current_day_nodes = []
                
                data = dict()
                
                G = nx.DiGraph()
                
                for k in range(len(edges)):
                    G.add_edge(edges[k][0], edges[k][1], weight = lengths[k], n_mutations = n_mutations[k], hop = 0.5)
                
                # find the nodes which have no out degree
                for node in G.nodes():
                    d = G.out_degree(node)
                    
                    if d == 0:
                        current_day_nodes.append(node)
                        
                        
                for node in current_day_nodes[:len(current_day_nodes) // 2]:
                    data[node] = [0., 1., 0., 0.]
                
                for node in current_day_nodes[len(current_day_nodes) // 2:]:
                    data[node] = [0., 0., 1., 0.]
                    
                t = 0.
                
                nodes = copy.copy(current_day_nodes)
                while len(data.keys()) < len(G.nodes()):
                    _ = []
                    for node in nodes:
                        for j in range(len(edges)):
                            if edges[j][-1] == node:
                                p = edges[j][0]
                                break
                        
                        data[p] = [data[node][0] + lengths[j], 0., 0., 1.]
                        _.append(p)
                        
                    nodes = copy.copy(_)
                    
                X = []
                for node in sorted(data.keys()):
                    X.append(data[node])
                    
                X = np.array(X)
                X[:,0] /= np.max(X[:,0])
                
                G = nx.Graph()
                
                for k in range(len(edges)):
                    G.add_edge(edges[k][0], edges[k][1], weight = lengths[k], n_mutations = n_mutations[k], hop = 0.5)
                    #G.add_edge(edges[k][1], edges[k][0], weight = lengths[k], n_mutations = n_mutations[k], hop = 0.5)
                
                t1 = time.time()
                paths = dict(nx.all_pairs_shortest_paths(G))
                
                nodes = list(range(0, 300))
                
                indices = list(itertools.combinations(nodes, 2))
                
                D = np.array([len(paths[i][j]) for (i,j) in indices])
                #logging.info('{}'.format(np.where(D == 0)))
                
                edges = np.array(edges).T
                
                comm.send([ix, ij, edges, X, regions, D], dest = 0)
                
            comm.send([ix, snps], dest = 0)
    
        # done signal
        comm.send([None], dest = 0)
    
    else:
        n_done = 0
        index = dict()
        
        sims_done = 0
        
        while n_done < comm.size - 1:
            v = comm.recv(source = MPI.ANY_SOURCE)
            
            if len(v) < 2:
                n_done += 1
                
                print('have {} processes done...'.format(n_done))
                continue
        
            if v[0] in index.keys():
                ii = index[v[0]]
            else:
                index[v[0]] = counter
                counter += 1
                
                ii = index[v[0]]
                
                print('seen {0} simulations out of {1}...'.format(np.max(list(index.values())), len(anc_files)))
              
            if len(v) == 7:
                ix, ij, edges, X, regions, n_mutations, D = v
                
                ofile.create_dataset('{0}/graph/{1}/xg'.format(ii, ij), data = X, compression = 'lzf') # time, population
                ofile.create_dataset('{0}/graph/{1}/edge_index'.format(ii, ij), data = edges.astype(np.int32), compression = 'lzf') # indices of branches (i.e. graph edges as index pairs)
                ofile.create_dataset('{0}/graph/{1}/regions'.format(ii, ij), data = np.array(regions).astype(np.int32), compression = 'lzf')
                ofile.create_dataset('{0}/graph/{1}/n_mutations'.format(ii, ij), data = np.array(n_mutations), compression = 'lzf')
                ofile.create_dataset('{0}/graph/{1}/D'.format(ii, ij), data = np.array(D, dtype = np.float32), compression = 'lzf')
                #ofile.create_dataset('{0}/graph/{1}/N'.format(ii, ij), data = np.array(N, dtype = np.float32), compression = 'lzf')
                #ofile.create_dataset('{0}/graph/{1}/Nh'.format(ii, ij), data = np.array(Nh, dtype = np.float32), compression = 'lzf')

            elif len(v) == 4:
                _, x, y, pos = v
                
                ofile.create_dataset('{0}/x'.format(ii), data = x.astype(np.uint8), compression = 'lzf')
                ofile.create_dataset('{0}/y'.format(ii), data = y.astype(np.uint8), compression = 'lzf')
                ofile.create_dataset('{0}/positions'.format(ii), data = pos, compression = 'lzf')
            elif len(v) == 2:
                _, snps = v
                
                ofile.create_dataset('{0}/break_points'.format(ii), data = np.array(snps, dtype = np.int32), compression = 'lzf')
                sims_done += 1
                
                print('wrote {0} simulations out of {1}...'.format(sims_done, len(anc_files)))
            
            ofile.flush()
                    
                
    if comm.rank == 0:     
        ofile.close()
    
    # ${code_blocks}

if __name__ == '__main__':
    main()

