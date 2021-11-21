# -*- coding: utf-8 -*-
import os
import argparse
import logging

import networkx as nx

import sys
import copy
import h5py
import numpy as np

import matplotlib
matplotlib.use('Agg')



# use this format to tell the parsers
# where to insert certain parts of the script
# ${imports}

def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser()
    # my args
    parser.add_argument("--verbose", action = "store_true", help = "display messages")
    parser.add_argument("--idir", default = "None")

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
    
    idirs_relate = sorted([os.path.join(args.idir, u) for u in os.listdir(args.idir) if 'relate' in u])
    idirs = sorted([os.path.join(args.idir, u) for u in os.listdir(args.idir) if not ('relate' in u)])
    
    ofile = h5py.File(args.ofile, 'w')
    
    dnsps = []
    
    for ii in range(len(idirs_relate)):
        idir = idirs[ii]
        idir_relate = idirs_relate[ii]
        
        print('working on {}...'.format(idir))
        
        anc_files = [os.path.join(idir_relate, u) for u in os.listdir(idir_relate) if u.split('.')[-1] == 'anc']
        
        for ix in range(len(anc_files)):
            ifile = os.path.join(idir, '{}.npz'.format(anc_files[ix].split('/')[-1].split('.')[0]))
            ifile = np.load(ifile)
            
            x = ifile['x']
            y = ifile['y']
            pos = ifile['positions']
            
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
                
                for j in range(2, len(line), 5):
                    nodes.append((j - 1) // 5)
                    
                    parents.append(int(line[j]))
                    lengths.append(float(line[j + 1]))
                    n_mutations.append(float(line[j + 2]))
                    regions.append((int(line[j + 3]), int(line[j + 4])))
                    
                    edges.append((parents[-1], nodes[-1]))
                    
                G = nx.DiGraph()
                G.add_edges_from(edges)
                
                current_day_nodes = []
                
                data = dict()
                
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
                
                edges = np.array(edges).T
            
                ofile.create_dataset('{2}/graph/{0}/{1}/xg'.format(ix, ij, ii), data = X, compression = 'lzf')
                ofile.create_dataset('{2}/graph/{0}/{1}/edge_index'.format(ix, ij, ii), data = edges.astype(np.int32), compression = 'lzf')
                ofile.create_dataset('{2}/graph/{0}/{1}/regions'.format(ix, ij, ii), data = np.array(regions).astype(np.int32), compression = 'lzf')
                ofile.create_dataset('{2}/graph/{0}/{1}/n_mutations'.format(ix, ij, ii), data = np.array(n_mutations), compression = 'lzf')
                
            
            ofile.create_dataset('{0}/x'.format(ii), data = x.astype(np.uint8), compression = 'lzf')
            ofile.create_dataset('{0}/y'.format(ii), data = y.astype(np.unit8), compression = 'lzf')
            ofile.create_dataset('{0}/positions'.format(ii), data = pos, compresssion = 'lzf')
            ofile.create_dataset('{0}/break_points'.format(ii), data = np.array(snps, dtype = np.int32), compression = 'lzf')
            
            ofile.flush()

    
                
                
            
    ofile.close()
    
    # ${code_blocks}

if __name__ == '__main__':
    main()

