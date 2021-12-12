# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
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
from scipy.spatial.distance import squareform

import time
import itertools
import random

from scipy.ndimage import gaussian_filter

EDGE_MEAN = np.array([
       8.9814079e-01, 1.0996692e+00, 1.0816846e+03, 1.5813326e+01],
      dtype=np.float32)
EDGE_STD = np.array([5.2765501e-01, 8.9487451e-01, 1.1330474e+03, 2.4047951e+01])

from mpi4py import MPI

# use this format to tell the parsers
# where to insert certain parts of the script
# ${imports}

def format_example(ifile, key, nn_samp, n_samples, n_sites = 128):
    X = []
    Y = []
    edge_attrs = []
    edge_indexes = []
    masks = []
    
    x_ = np.array(ifile[key]['x'])
    y_ = np.array(ifile[key]['y'])
    
    bp = np.array(ifile[key]['break_points'])
    if len(bp) <= 1:
        return None
    
    for j in range(n_samples):
        bp_ix = np.random.choice(range(len(bp) - 1))
        
        bp0 = bp[bp_ix]
        bp1 = bp[bp_ix + 1]
        
        if bp1 - bp0 < n_sites:
        
            x = x_[:,bp0:bp1]
            y = y_[:,bp0:bp1]
        else:
            x = x_[:,bp0:bp0 + n_sites]
            y = y_[:,bp0:bp0 + n_sites]
        
        x = np.pad(x, ((0, 0), (0, n_sites - x.shape[1])), constant_values = -1)
        x = np.pad(x, ((0, 299), (0, 0)), constant_values = 0)
        
        y_mask = np.ones(y.shape)
        dpad =  n_sites - y.shape[1]
        
        y = np.pad(y, ((0, 0), (0, n_sites - y.shape[1])), constant_values = -1)
        y = np.pad(y, ((0, 299), (0, 0)), constant_values = 0)
        
        y_mask = np.pad(y_mask, ((0, 0), (0, dpad)), constant_values = 0)
        y_mask = np.pad(y_mask, ((0, 299), (0, 0)), constant_values = 0)
        y_mask[:150,:] = 0
        
        edges = np.array(ifile[key]['graph']['{}'.format(bp_ix)]['edge_index'])[:,:-1]
        xg = np.array(ifile[key]['graph']['{}'.format(bp_ix)]['xg'])[:-1,:]
        n_mutations = np.array(ifile[key]['graph']['{}'.format(bp_ix)]['n_mutations'])[:-1]
        n_mutations = n_mutations.reshape(n_mutations.shape[0], 1)
        
        edge_attr = np.concatenate([xg[edges[0,:]] - xg[edges[1,:]], n_mutations], axis = 1)
        
        X.append(x)
        edge_indexes.append(edges)
        edge_attrs.append(edge_attr)
        masks.append(y_mask)
        
    return X, Y, edge_indexes, edge_attrs, masks
        
def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser()
    # my args
    parser.add_argument("--verbose", action = "store_true", help = "display messages")
    parser.add_argument("--idir", default = "None")
    
    parser.add_argument("--k", default = "3")
    parser.add_argument("--chunk_size", default = "4")
    parser.add_argument("--n_samples", default = "10")
    parser.add_argument("--n_sites", default = "64")
    
    parser.add_argument("--val_prop", default = 0.05)

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
    
    ifiles = [os.path.join(args.idir, u) for u in os.listdir(args.idir) if u.split('.')[-1] == 'hdf5']
    chunk_size = 4
    
    nn_samp = range(int(args.k))
    n_sites = int(args.n_sites)
    
    train_counter = 0
    val_counter = 0
    
    comm.Barrier()
    
    if comm.rank != 0:
        for ix in range(comm.rank - 1, len(ifiles), comm.size - 1):
            logging.info('{0}: on file {1}...'.format(comm.rank, ifiles[ix]))
            
            ifile = h5py.File(ifiles[ix], 'r')
            keys = list(ifile.keys())
            
            random.shuffle(keys)
            n_val = int(np.round(len(keys) * float(args.val_prop)))
            
            val_keys = keys[:n_val]
            del keys[:n_val]
            
            for key in keys:
                _ = format_example(ifile, key, nn_samp, int(args.n_samples), n_sites)
                if _ is None:
                    continue
                
                x, y, edge_index, edge_attr, masks = _

                
                comm.send([x, y, edge_index, edge_attr, masks, False], dest = 0)
                
            for key in val_keys:
                _ = format_example(ifile, key, nn_samp, int(args.n_samples), n_sites)
                if _ is None:
                    continue
                
                x, y, edge_index, edge_attr, masks = _
                
                comm.send([x, y, edge_index, edge_attr, masks, True], dest = 0)
                
        comm.send([None, None, None, None, None, None], dest = 0)
    
    else:
        x = []
        y = []
        edge_attr = []
        edge_index = []
        mask = []
        
        x_val = []
        y_val = []
        edge_attr_val = []
        edge_index_val = []
        mask_val = []
        
        n_done = 0
        train_counter = 0
        val_counter = 0
        
        while n_done < comm.size - 1:
            x_, y_, edge_index_, edge_attr_, mask_, val = comm.recv(source = MPI.ANY_SOURCE)
            
            if x_ is None:
                n_done += 1
                continue
            
            if not val:
                x.extend(x_)
                y.extend(y_)
                edge_attr.extend(edge_attr_)
                edge_index.extend(edge_index_)
                mask.extend(mask_)
            # it's validation data
            else:
                x_val.extend(x_)
                y_val.extend(y_)
                edge_attr_val.extend(edge_attr_)
                edge_index_val.extend(edge_index_)
                mask_val.extend(mask_)
                
            # maybe write some data
            if (len(x) % chunk_size) == 0 and len(x) > 0:
                ix = list(range(len(x)))
                random.shuffle(ix)
                
                ix = list(chunks(ix, chunk_size))
            
                for ix_ in ix:
                    x_ = np.array([x[u] for u in ix_], dtype = np.float32)
                    y_ = np.array([y[u] for u in ix_], dtype = np.uint8)
                    edge_index_ = np.array([edge_index[u] for u in ix_], dtype = np.int32)
                    edge_attr_ = np.array([edge_attr[u] for u in ix_], dtype = np.float32)
                    mask_ = np.array([mask[u] for u in ix_], dtype = np.uint8)
                    
                    ofile.create_dataset('train/{0}/x_0'.format(train_counter), data = x_, compression = 'lzf')
                    ofile.create_dataset('train/{0}/y'.format(train_counter), data = y_, compression = 'lzf')
                    ofile.create_dataset('train/{0}/edge_index'.format(train_counter), data = edge_index_, compression = 'lzf')
                    ofile.create_dataset('train/{0}/edge_attr'.format(train_counter), data = edge_attr_, compression = 'lzf')
                    ofile.create_dataset('train/{0}/mask'.format(train_counter), data = mask_, compression = 'lzf')
                    
                    if (train_counter + 1) % 25 == 0: 
                        logging.info('0: wrote chunk {}...'.format(train_counter))
                    train_counter += 1
                    
                    ofile.flush()
                    
                x = []
                y = []
                edge_attr = []
                edge_index = []
                
            # maybe write some data
            if (len(x_val) % chunk_size) == 0 and len(x_val) > 0:
                ix = list(range(len(x_val)))
                random.shuffle(ix)
                
                ix = list(chunks(ix, chunk_size))
            
                for ix_ in ix:
                    x_ = np.array([x_val[u] for u in ix_], dtype = np.float32)
                    y_ = np.array([y_val[u] for u in ix_], dtype = np.uint8)
                    edge_index_ = np.array([edge_index_val[u] for u in ix_], dtype = np.int32)
                    edge_attr_ = np.array([edge_attr_val[u] for u in ix_], dtype = np.float32)
                    mask_ = np.array([mask_val[u] for u in ix_], dtype = np.uint8)
                    
                    ofile.create_dataset('val/{0}/x_0'.format(val_counter), data = x_, compression = 'lzf')
                    ofile.create_dataset('val/{0}/y'.format(val_counter), data = y_, compression = 'lzf')
                    ofile.create_dataset('val/{0}/edge_index'.format(val_counter), data = edge_index_, compression = 'lzf')
                    ofile.create_dataset('val/{0}/edge_attr'.format(val_counter), data = edge_attr_, compression = 'lzf')
                    ofile.create_dataset('train/{0}/mask'.format(train_counter), data = mask_, compression = 'lzf')

                    if (val_counter + 1) % 25 == 0: 
                        logging.info('0: wrote val chunk {}...'.format(val_counter))
                    val_counter += 1
                    
                    ofile.flush()
                    
                x_val = []
                y_val = []
                edge_attr_val = []
                edge_index_val = []
    
    if comm.rank == 0:
        ofile.close()
            
if __name__ == '__main__':
    main()
                
            
            
        
        

