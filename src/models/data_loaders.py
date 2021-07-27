# -*- coding: utf-8 -*-
import h5py
import numpy as np

from collections import deque
import copy
import random

import torch
import os, sys

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
                
            X1.extend(list(np.array(self.ifile['train'][k]['x1'])))
            X2.extend(list(np.array(self.ifile['train'][k]['x2'])))
            
            y.extend([0 for u in range(X1[-1].shape[0])])
            
            # real data
            X = self.Xs_val[np.random.choice(range(len(self.Xs_val)))]
            
            k = np.random.choice(range(X.shape[1] - X1[-1].shape[-1]), 4, replace = False)
            
            for ii in k:
                X1.append(X[:20, ii: ii + X1[-1].shape[-1]])
                X2.append(X[20:, ii: ii + X1[-1].shape[-1]])
                
                y.append(1)
            
        return torch.FloatTensor(np.array(X1).reshape(len(X1), 1, X1[0].shape[0], X1[0].shape[1])), torch.FloatTensor(np.array(X2).reshape(len(X1), 1, X2[0].shape[0], X1[0].shape[1])), torch.LongTensor(y)
    
if __name__ == '__main__':
    ifile = sys.argv[1]
    idir = sys.argv[2]
    
    print('instantiatig object...')
    gen = H5DisDataGenerator(ifile, idir)
    
    print('getting batch...')
    x1, x2, y = gen.get_batch()
    
    print(x1.shape, x2.shape, y.shape)
    print(y)
    
    
    