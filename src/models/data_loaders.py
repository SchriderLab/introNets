# -*- coding: utf-8 -*-
import h5py
import numpy as np

from collections import deque
import copy
import random

import torch
import os, sys

import glob

from scipy.spatial.distance import squareform

from data_functions import load_data_dros
import pickle
from sklearn.neighbors import kneighbors_graph

class H5DisDataGenerator(object):
    def __init__(self, ifiles, n_samples = 50000, n_samples_val = 1000, chunk_size = 4, batch_size = 32):
        # ifiles is a dictionary pointing to the file names where h5 files corresponding to the class are
        # how many classes / keys?
        self.n_classes = len(ifiles.keys())
        self.classes = sorted(ifiles.keys())
        
        # make a dictionary of the file objects to read
        self.ifiles = dict(zip(self.classes, [[h5py.File(ifiles[u][k], 'r') for k in range(len(ifiles[u]))] for u in self.classes]))
        
        self.key_dict = dict(zip(self.classes, [[list(self.ifiles[u][k].keys()) for k in range(len(ifiles[u]))] for u in self.classes]))
        
        self.train_keys = dict()
        self.val_keys = dict()
        
        for c in self.classes:
            self.train_keys[c] = []
            self.val_keys[c] = []
            
            for k in range(len(self.key_dict[c])):
                random.shuffle(self.key_dict[c][k])
                
                self.train_keys[c].extend([(k, u) for u in self.key_dict[c][k][:n_samples // len(self.key_dict[c])]])
                self.val_keys[c].extend([(k,u) for u in self.key_dict[c][k][n_samples // len(self.key_dict[c]):n_samples // len(self.key_dict[c]) + n_samples_val // len(self.key_dict[c])]])
                
            random.shuffle(self.train_keys[c])
            
        self.n_per_class = (batch_size // chunk_size) // self.n_classes
        
        self.length = min([len(self.train_keys[u]) // self.n_per_class for u in self.classes])
        self.val_length = min([len(self.val_keys[u]) // self.n_per_class for u in self.classes])
        
        self.ix = 0
        self.ix_val = 0
        
    def on_epoch_end(self):
        self.ix = 0
        self.ix_val = 0
        
        for c in self.classes:
            random.shuffle(self.train_keys[c])
        
    def get_batch(self, val = False):
        X = []
        Y = []
        
        for c in self.classes:
            if not val:
                keys = self.train_keys[c][self.ix*self.n_per_class : (self.ix + 1)*self.n_per_class]
                self.ix += 1
            else:
                keys = self.val_keys[c][self.ix_val*self.n_per_class : (self.ix_val + 1)*self.n_per_class]
                self.ix_val += 1
            
            for k, u in keys:
                x = np.array(self.ifiles[c][k][u]['x_0'], dtype = np.float32)

                Y.extend([self.classes.index(c) for j in range(x.shape[0])])
                X.append(x)
                
        if len(X) == 0:
            return None, None
        
        X = np.vstack(X)
        
        if val:
            self.ix_val += 1
        else:
            self.ix += 1
        
        return torch.FloatTensor(X), torch.LongTensor(Y)
    
    def get_batch_dual(self, val = False):
        X1 = []
        X2 = []
        Y = []
        
        for c in self.classes:
            if not val:
                keys = self.train_keys[c][self.ix*self.n_per_class : (self.ix + 1)*self.n_per_class]
                self.ix += 1
            else:
                keys = self.val_keys[c][self.ix_val*self.n_per_class : (self.ix_val + 1)*self.n_per_class]
                self.ix_val += 1
            
            for k, u in keys:
                x1 = np.array(self.ifiles[c][k][u]['x2'], dtype = np.float32)
                x2 = np.array(self.ifiles[c][k][u]['x1'], dtype = np.float32)
                
                Y.extend([self.classes.index(c) for j in range(x1.shape[0])])
                X1.append(np.expand_dims(x1, axis = 1))
                X2.append(np.expand_dims(x2, axis = 1))
                
        if len(X1) == 0:
            return None, None, None
        
        X1 = np.vstack(X1)
        X2 = np.vstack(X2)
        
        if val:
            self.ix_val += 1
        else:
            self.ix += 1
        
        return torch.FloatTensor(X1), torch.FloatTensor(X2), torch.LongTensor(Y)

class H5UDataGenerator(object):
    def __init__(self, ifile, keys = None, 
                 val_prop = 0.05, batch_size = 16, 
                 chunk_size = 4, pred_pop = 1, label_noise = 0.01, label_smooth = True):
        if keys is None:
            self.keys = list(ifile.keys())
            
            n_val = int(len(self.keys) * val_prop)
            random.shuffle(self.keys)
            
            self.val_keys = self.keys[:n_val]
            del self.keys[:n_val]
            
        self.ifile = ifile
        
        self.chunk_size = chunk_size
        self.batch_size = batch_size
        self.label_noise = label_noise
        self.label_smooth = label_smooth
            
        self.length = len(self.keys) // (batch_size // chunk_size)
        self.val_length = len(self.val_keys) // (batch_size // chunk_size)
        
        self.n_per = batch_size // chunk_size
        
        self.pred_pop = pred_pop
        
        self.ix = 0
        self.ix_val = 0
            
    def define_lengths(self):
        self.length = len(self.keys) // (self.batch_size // self.chunk_size)
        self.val_length = len(self.val_keys) // (self.batch_size // self.chunk_size)
        
    def get_batch(self):
        X = []
        Y = []
        
        for key in self.keys[self.ix : self.ix + self.n_per]:
            x = np.array(self.ifile[key]['x_0'])
            y = np.array(self.ifile[key]['y'])
            
            X.append(x)
            Y.append(y)
            
        Y = np.concatenate(Y)
        
        if self.label_smooth:
            # label smooth
            ey = np.random.uniform(0, self.label_noise, Y.shape)
            
            Y = Y * (1 - ey) + 0.5 * ey
            
        self.ix += self.n_per
        return torch.FloatTensor(np.concatenate(X)), torch.FloatTensor(Y)
    
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
            
        Y = np.concatenate(Y)
            
        self.ix_val += self.n_per
        return torch.FloatTensor(np.concatenate(X)), torch.FloatTensor(Y)
    
    
    
    