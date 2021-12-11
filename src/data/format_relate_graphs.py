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
    x = []
    y = []
    edge_attr = []
    edge_index = []
    
    x_ = np.array(ifile[key]['x'])
    y_ = np.array(ifile[key]['y'])
    
    bp = np.array(ifile[key]['break_points'])
    if len(bp) == 0:
        return None, None, None, None
    
    indices = range(bp[0], bp[-1])
    for j in range(n_samples):
        D = np.zeros((4, 300, 300))
        count = 0
        
        s = np.random.choice(range(indices[0], indices[-1] - n_sites))
        s = list(range(s, s + n_sites))
        
        gix = list(np.where((bp >= s[0]) & (bp < s[-1]))[0])
        
        if len(gix) == 0:
            continue
        
        # currently a weighted average of the distance matrices in the region
        for k in gix:
            D_ = np.array(ifile[key]['graph']['{}'.format(k)]['D'])
            
            D_ = np.array([squareform(u) for u in D_], dtype = np.float32)
            
            if bp[k + 1] <= s[-1]:
                w = (bp[k + 1] - bp[k]) / n_sites
            else:
                w = (s[-1] - bp[k]) / n_sites
        
            D += D_ * w
            count += 1
        
        D /= count
        
        # hop, bran
        # preserve n_mutations as the un-scaled one
        D[1,:,:] *= count
        
        x_e = x_[:,s]
        y_e = y_[150:,s]
        
        bp_ = [u for u in bp if u in s]
        bp_ = list(np.array(bp_) - int(np.min(bp_)))

        edge_index_ = []
        edge_attr_ = []
        
        # from pop 1
        for ix in range(D.shape[1] // 2):
            # to pop 1
            ij1 = np.argsort(D[0, ix, :D.shape[1] // 2])[nn_samp]
            ij2 = np.argsort(D[0, ix, D.shape[1] // 2:])[nn_samp]
            
            # pop 1 -> 2
            edge_class = np.zeros((len(nn_samp), 4))
            edge_class[:,2] = 1
            
            # pop 1 -> 1
            edge_class_ = np.zeros((len(nn_samp), 4))
            edge_class_[:,0] = 1
            
            edge_class = np.concatenate([edge_class, edge_class_], axis = 0)
            
            ij = np.array(list(ij1) + list(ij2))
            
            _ = np.vstack([(D[:,ix,u] - EDGE_MEAN) / EDGE_STD for u in ij])
        
            ## i -> j
            edge_index_.extend([(ix, u) for u in ij])
            edge_attr_.extend(list(np.concatenate([edge_class, _], axis = 1)))
           
            
        # from pop 2
        for ix in range(D.shape[0] // 2, D.shape[1]):
            # to pop 1
            ij1 = np.argsort(D[0, ix, :D.shape[1] // 2])[nn_samp]
            # to pop 2
            ij2 = np.argsort(D[0, ix, D.shape[1] // 2:])[nn_samp]
            
            # pop 2 -> 2
            edge_class = np.zeros((len(nn_samp), 4))
            edge_class[:,3] = 1
            
            # pop 2 -> 1
            edge_class_ = np.zeros((len(nn_samp), 4))
            edge_class_[:,1] = 1
            
            edge_class = np.concatenate([edge_class, edge_class_], axis = 0)
            
            ij = np.array(list(ij1) + list(ij2))
            
            _ = np.vstack([(D[:,u,ix] - EDGE_MEAN) / EDGE_STD for u in ij])
            
            ## i -> j
            edge_index_.extend([(u, ix) for u in ij])
            edge_attr_.extend(list(np.concatenate([edge_class, _], axis = 1)))
        
        edge_index_ = np.array(edge_index_, dtype = np.int32)
        edge_attr_ = np.array(edge_attr_, dtype = np.float32)
        
        #np.savez('edge_attrs/edge_attr_{0:04d}.npz'.format(self.ix), edge_attr = edge_attr)
        
        
        bp_x = np.zeros(x_e.shape)
        bp_x[:, bp_] = 1.
        
        # smooth the break points (give some error)
        bp_x = gaussian_filter(bp_x, 5)
        
        x_e = np.array((x_e, bp_x), dtype = np.float32)
        
        x.append(x_e)
        y.append(y_e)
        edge_index.append(edge_index_)
        edge_attr.append(edge_attr_)
        
    return x, y, edge_index, edge_attr
        
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
    parser.add_argument("--n_sites", default = "128")
    
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
    
    ifiles = [os.path.join(args.idir, u) for u in os.listdir(args.idir)]
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
                x, y, edge_index, edge_attr = format_example(ifile, key, nn_samp, int(args.n_samples), n_sites)
                if x is None:
                    continue
                
                comm.send([x, y, edge_index, edge_attr, False], dest = 0)
                
            for key in val_keys:
                x, y, edge_index, edge_attr = format_example(ifile, key, nn_samp, int(args.n_samples), n_sites)
                if x is None:
                    continue
                
                comm.send([x, y, edge_index, edge_attr, True], dest = 0)
                
        comm.send([None, None, None, None, None], dest = 0)
    
    else:
        x = []
        y = []
        edge_attr = []
        edge_index = []
        
        x_val = []
        y_val = []
        edge_attr_val = []
        edge_index_val = []
        
        n_done = 0
        train_counter = 0
        val_counter = 0
        
        while n_done < comm.size - 1:
            x_, y_, edge_index_, edge_attr_, val = comm.recv(source = MPI.ANY_SOURCE)
            
            if x_ is None:
                n_done += 1
                continue
            
            if not val:
                x.extend(x_)
                y.extend(y_)
                edge_attr.extend(edge_attr_)
                edge_index.extend(edge_index_)
            # it's validation data
            else:
                x_val.extend(x_)
                y_val.extend(y_)
                edge_attr_val.extend(edge_attr_)
                edge_index_val.extend(edge_index_)
                
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
                    
                    ofile.create_dataset('train/{0}/x_0'.format(train_counter), data = x_, compression = 'lzf')
                    ofile.create_dataset('train/{0}/y'.format(train_counter), data = y_, compression = 'lzf')
                    ofile.create_dataset('train/{0}/edge_index'.format(train_counter), data = edge_index_, compression = 'lzf')
                    ofile.create_dataset('train/{0}/edge_attr'.format(train_counter), data = edge_attr_, compression = 'lzf')
                    
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
                    
                    ofile.create_dataset('val/{0}/x_0'.format(val_counter), data = x_, compression = 'lzf')
                    ofile.create_dataset('val/{0}/y'.format(val_counter), data = y_, compression = 'lzf')
                    ofile.create_dataset('val/{0}/edge_index'.format(val_counter), data = edge_index_, compression = 'lzf')
                    ofile.create_dataset('val/{0}/edge_attr'.format(val_counter), data = edge_attr_, compression = 'lzf')
                    
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
                
            
            
        
        