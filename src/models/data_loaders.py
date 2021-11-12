# -*- coding: utf-8 -*-
import h5py
import numpy as np

from collections import deque
import copy
import random

import torch
import os, sys

from torch.utils.data import Dataset
from torch_geometric.data import Data
import glob
from sparsenn.models.gcn.topologies import knn_1d

from data_functions import load_data_dros
import pickle

def load_npz(ifile):
    ifile = np.load(ifile)
    pop1_x = ifile['simMatrix'].T
    pop2_x = ifile['sechMatrix'].T

    x = np.vstack((pop1_x, pop2_x))
    
    # destroy the perfect information regarding
    # which allele is the ancestral one
    for k in range(x.shape[1]):
        if np.sum(x[:,k]) > 17:
            x[:,k] = 1 - x[:,k]
        elif np.sum(x[:,k]) == 17:
            if np.random.choice([0, 1]) == 0:
                x[:,k] = 1 - x[:,k]

    return x

class GCNDisDataGenerator(object):
    def __init__(self, idir, batch_size = 8, 
                     val_prop = 0.05, k = 8, 
                     seg = False):
       
        self.training = glob.glob(os.path.join(idir, '*/*/*.npz'))
        self.models = sorted(list(set([u.split('/')[-3] for u in self.training])))
                
        n_val = int(len(self.training) * val_prop)
        random.shuffle(self.training)
        
        self.val = self.training[:n_val]
        del self.training[:n_val]
        
        self.batch_size = batch_size
        
        self.on_epoch_end()
        self.k = k
        self.seg = seg
        
        self.length = len(self.training) // self.batch_size
        self.val_length = len(self.val) // self.batch_size
                
    def on_epoch_end(self):
        random.shuffle(self.training)
        
        self.ix = 0
        self.val_ix = 0
        
    def get_element(self, val = False):
        if val:
            ifile = np.load(self.val[self.val_ix], allow_pickle = True)
            
            model = self.val[self.val_ix].split('/')[-3]
            self.val_ix += 1
        else:
            ifile = np.load(self.training[self.ix], allow_pickle = True)
            
            model = self.training[self.ix].split('/')[-3]
            self.ix += 1
                
        if len(ifile['y'].shape) == 0:
            return self.get_element(val)
        
        
        
        x = torch.FloatTensor(ifile['x'].T)
        
        edges = [torch.LongTensor(u) for u in ifile['edges']]
        
        y = torch.LongTensor([self.models.index(model)])
        
        return x, y, edges
        
    def get_batch(self, val = False):
        xs = []
        ys = []
        edges = []
        batch = []
        
        current_node = 0
        for ix in range(self.batch_size):
            x, y, e = self.get_element(val)
            
            n = x.shape[0]
            
            if len(edges) == 0:
                edges = e
            else:
                edges = [torch.cat([edges[k], e[k] + current_node], dim = 1) for k in range(len(e))]
                
            batch.extend(np.repeat(ix, n))
    
            xs.append(x)
            ys.append(y)
            
            current_node += n
            
        x = torch.cat(xs)
        y = torch.cat(ys)
        
        return x, y, edges, torch.LongTensor(batch)
    
    def __len__(self):
        return self.length

class GCNDataGenerator(object):
    def __init__(self, idir, indices, batch_size = 8, 
                     val_prop = 0.05, k = 8, 
                     seg = False):
       
        
        if indices == "None":
            self.training = glob.glob(os.path.join(idir, '*/*.npz')) + glob.glob(os.path.join(idir, '*.npz'))
                    
            n_val = int(len(self.training) * val_prop)
            random.shuffle(self.training)
            
            self.val = self.training[:n_val]
            del self.training[:n_val]
        else:
            self.training, self.val = pickle.load(open(indices, 'rb'))
            
            self.training = [os.path.join(idir, u) for u in self.training]
            self.val = [os.path.join(idir, u) for u in self.val]
        
        self.batch_size = batch_size
        
        self.on_epoch_end()
        self.k = k
        self.seg = seg
        
        self.length = len(self.training) // self.batch_size
        self.val_length = len(self.val) // self.batch_size
                
    def on_epoch_end(self):
        random.shuffle(self.training)
        
        self.ix = 0
        self.val_ix = 0
        
    def get_element(self, val = False):
        if val:
            if self.val_ix >= len(self.val):
                return None, None, None
            
            ifile = np.load(self.val[self.val_ix], allow_pickle = True)
            self.val_ix += 1
        else:
            if self.ix >= len(self.training):
                return None, None, None
            
            ifile = np.load(self.training[self.ix], allow_pickle = True)
            self.ix += 1
            
        if len(ifile['y'].shape) == 0:
            return self.get_element(val)
        
        if not self.seg:
            y = np.mean(ifile['y'].T, axis = 1)
            y = torch.FloatTensor(y.reshape(y.shape[0], 1))
        else:
            y = torch.FloatTensor(ifile['y'].T)
        
        x = torch.FloatTensor(ifile['x'].T)
        
        if x.shape[0] != y.shape[0]:
            return self.get_element(val)
        
        edges = [torch.LongTensor(u) for u in ifile['edges']]
        
        return x, y, edges
        
    def get_batch(self, val = False):
        xs = []
        ys = []
        edges = []
        batch = []
        
        current_node = 0
        for ix in range(self.batch_size):
            x, y, e = self.get_element(val)
            
            if x is None:
                break
            
            n = x.shape[0]
            
            if len(edges) == 0:
                edges = e
            else:
                edges = [torch.cat([edges[k], e[k] + current_node], dim = 1) for k in range(len(e))]
                
            batch.extend(np.repeat(ix, n))
    
            xs.append(x)
            ys.append(y)
            
            current_node += n
            
        x = torch.cat(xs)
        y = torch.cat(ys)
        
        if x.shape[0] != y.shape[0]:
            print('somethings wrong...')
        
        return x, y, edges, torch.LongTensor(batch)
    
    def __len__(self):
        return self.length

class H5UDataGenerator(object):
    def __init__(self, ifile, keys = None, val_prop = 0.05, batch_size = 16, chunk_size = 4):
        if keys is None:
            self.keys = list(ifile.keys())
            
            n_val = int(len(self.keys) * val_prop)
            random.shuffle(self.keys)
            
            self.val_keys = self.keys[:n_val]
            del self.keys[:n_val]
            
        self.ifile = ifile
            
        self.length = len(self.keys) // (batch_size // chunk_size)
        self.val_length = len(self.val_keys) // (batch_size // chunk_size)
        
        self.n_per = batch_size // chunk_size
        
        self.ix = 0
        self.ix_val = 0
            
    def get_batch(self):
        X = []
        Y = []
        
        for key in self.keys[self.ix : self.ix + self.n_per]:
            x = np.array(self.ifile[key]['x_0'])
            y = np.array(self.ifile[key]['y'])
            
            X.append(x)
            Y.append(y)
            
        self.ix += self.n_per
        return torch.FloatTensor(np.concatenate(X)), torch.FloatTensor(np.concatenate(Y))
    
    def on_epoch_end(self):
        self.ix = 0
        self.ix_val = 0
        
        random.shuffle(self.keys)
        
    def get_val_batch(self):
        X = []
        Y = []
        
        for key in self.val_keys[self.ix_val : self.ix_val + self.n_per]:
            x = np.array(self.ifile[key]['x_0'])
            y = np.array(self.ifile[key]['y'])
            
            X.append(x)
            Y.append(y)
            
        self.ix_val += self.n_per
        return torch.FloatTensor(np.concatenate(X)), torch.FloatTensor(np.concatenate(Y))
        
class DisDataGenerator(object):
    def __init__(self, idir_sims, idir_real, batch_size = 64):
        self.Xr = [load_npz(os.path.join(idir_real, u)) for u in sorted(os.listdir(idir_real))]
        
        self.Xr_val = self.Xr[-3:]
        del self.Xr[-3:]
        
        self.Xs_val = [[], []]
        
        self.idirs = sorted([os.path.join(idir_sims, u) for u in os.listdir(idir_sims)])
        self.Xs = None
        
        self.ix_s = 0
        
        self.done = False
        self.done_val = False
        self.batch_size = batch_size
        
        self.on_epoch_end()
        
    def on_epoch_end(self):
        self.ix_s = 0
        self.Xs_val = [[], []]
        
        self.read()
        
        self.done = False
        self.done_val = False
        
    def read(self):
        ms = os.path.join(self.idirs[self.ix_s], 'mig.msOut')
        anc = os.path.join(self.idirs[self.ix_s], 'out.anc')
        
        try:
            x1, x2, y1, y2, params = load_data_dros(ms, anc)
        except:
            self.ix_s += 1
            
            if self.ix_s >= len(self.idirs):
                self.done = True
                return False
            else:
                return self.read()
        
        self.x1s = x1
        self.x2s = x2
        
        self.Xs_val[0].extend(self.x1s[-100:])
        self.Xs_val[1].extend(self.x2s[-100:])
        
        del self.x1s[-100:]
        del self.x2s[-100:]
        
        self.ix_s += 1
        
        if self.ix_s == len(self.idirs):
            self.done = True
            
        return True
        
        
    def get_batch(self):
        X1 = []
        X2 = []
        y = []

        if len(self.x1s) < self.batch_size // 2:
            if self.read():
                return self.get_batch()
            else:
                return None, None, None
            
        X1.extend([np.expand_dims(u, 0) for u in self.x1s[:self.batch_size // 2]])
        X2.extend([np.expand_dims(u, 0) for u in self.x2s[:self.batch_size // 2]])
        
        del self.x1s[:self.batch_size // 2]
        del self.x2s[:self.batch_size // 2]
        
        y.extend([0 for u in range(self.batch_size // 2)])
        
        # real data
        X = self.Xr[np.random.choice(range(len(self.Xr)))]
        
        k = np.random.choice(range(X.shape[1] - X1[-1].shape[-1]), self.batch_size // 2, replace = False)
            
        for ii in k:
            X1.append(np.expand_dims(X[:20, ii: ii + X1[-1].shape[-1]], axis = 0))
            X2.append(np.expand_dims(X[20:, ii: ii + X1[-1].shape[-1]], axis = 0))
            
            y.append(1)
            
        return torch.FloatTensor(np.expand_dims(np.concatenate(X1), axis = 1)), torch.FloatTensor(np.expand_dims(np.concatenate(X2), axis = 1)), torch.LongTensor(y)
            
    def get_val_batch(self):
        X1 = []
        X2 = []
        y = []
            
        X1.extend([np.expand_dims(u, 0) for u in self.Xs_val[0][:self.batch_size // 2]])
        X2.extend([np.expand_dims(u, 0) for u in self.Xs_val[1][:self.batch_size // 2]])
        
        del self.Xs_val[0][:self.batch_size // 2]
        del self.Xs_val[1][:self.batch_size // 2]
        
        if len(self.Xs_val[0]) < self.batch_size // 2:
            self.done_val = True
        
        y.extend([0 for u in range(self.batch_size // 2)])
        
        # real data
        X = self.Xr_val[np.random.choice(range(len(self.Xr_val)))]
        
        k = np.random.choice(range(X.shape[1] - X1[-1].shape[-1]), self.batch_size // 2, replace = False)
            
        for ii in k:
            X1.append(np.expand_dims(X[:20, ii: ii + X1[-1].shape[-1]], axis = 0))
            X2.append(np.expand_dims(X[20:, ii: ii + X1[-1].shape[-1]], axis = 0))
            
            y.append(1)
            
        return torch.FloatTensor(np.expand_dims(np.concatenate(X1), axis = 1)), torch.FloatTensor(np.expand_dims(np.concatenate(X2), axis = 1)), torch.LongTensor(y)
        

class H5DisDataGenerator(object):
    def __init__(self, ifile, idir, n_chunks = 8):
        self.ifile = h5py.File(ifile, 'r')
        self.Xs = [load_npz(os.path.join(idir, u)) for u in sorted(os.listdir(idir))]
        
        self.Xs_val = self.Xs[-3:]
        del self.Xs[-3:]
        
        self.keys = list(self.ifile['train'].keys())
        self.val_keys = list(self.ifile['val'].keys())        
        
        self.on_epoch_end()
    
        self.n_chunks = n_chunks
        return
    
    def on_epoch_end(self):
        self.keys_ = copy.copy(self.keys)
        random.shuffle(self.keys_)
        
        self.val_keys_ = copy.copy(self.val_keys)
        
        self.keys_ = deque(self.keys_)
        self.val_keys_ = deque(self.val_keys_)
        
        return
    
    def get_lengths(self):
        length = len(self.keys) // self.n_chunks
        val_length = len(self.val_keys) // self.n_chunks
        
        return length, val_length
    
    def get_batch(self):
        X1 = []
        X2 = []
        
        y = []
        
        for ix in range(self.n_chunks):
            k = self.keys_.pop()
        
            X1.append(np.array(self.ifile['train'][k]['x1']))
            X2.append(np.array(self.ifile['train'][k]['x2']))
            
            y.extend([0 for u in range(X1[-1].shape[0])])
            
            # real data
            X = self.Xs[np.random.choice(range(len(self.Xs)))]
            
            k = np.random.choice(range(X.shape[1] - X1[-1].shape[-1]), 4, replace = False)
            
            for ii in k:
                X1.append(np.expand_dims(X[:20, ii: ii + X1[-1].shape[-1]], axis = 0))
                X2.append(np.expand_dims(X[20:, ii: ii + X1[-1].shape[-1]], axis = 0))
                
                y.append(1) 
        
        return torch.FloatTensor(np.expand_dims(np.concatenate(X1), axis = 1)), torch.FloatTensor(np.expand_dims(np.concatenate(X2), axis = 1)), torch.LongTensor(y)
    
    def get_val_batch(self):
        X1 = []
        X2 = []
        
        y = []
        
        for ix in range(self.n_chunks):
            k = self.val_keys_.pop()
                
            X1.append(np.array(self.ifile['val'][k]['x1']))
            X2.append(np.array(self.ifile['val'][k]['x2']))
            
            y.extend([0 for u in range(X1[-1].shape[0])])
            
            # real data
            X = self.Xs_val[np.random.choice(range(len(self.Xs_val)))]
            
            k = np.random.choice(range(X.shape[1] - X1[-1].shape[-1]), 4, replace = False)
            
            for ii in k:
                X1.append(np.expand_dims(X[:20, ii: ii + X1[-1].shape[-1]], axis = 0))
                X2.append(np.expand_dims(X[20:, ii: ii + X1[-1].shape[-1]], axis = 0))
                
                y.append(1) 
        
        return torch.FloatTensor(np.expand_dims(np.concatenate(X1), axis = 1)), torch.FloatTensor(np.expand_dims(np.concatenate(X2), axis = 1)), torch.LongTensor(y)
    
if __name__ == '__main__':
    ifile = sys.argv[1]
    idir = sys.argv[2]
    
    print('instantiatig object...')
    gen = H5DisDataGenerator(ifile, idir)
    
    print('getting batch...')
    x1, x2, y = gen.get_batch()
    
    print(x1.shape, x2.shape, y.shape)
    print(y)
    
    
    