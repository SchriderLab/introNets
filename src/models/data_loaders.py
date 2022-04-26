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
    
    
    
    