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


class DGLH5DataGenerator(object):
    def __init__(self, ifile, batch_size = 16, chunk_size = 4):
        self.ifile = h5py.File(ifile, 'r')
        
        self.reset_keys()
        self.reset_keys(True)
        
        self.n_per = batch_size // chunk_size
        self.batch_size = batch_size
        
        self.val_length = len(self.val_keys) // self.n_per
        
    def reset_keys(self, val = False):
        if not val:
            self.train_keys = list(self.ifile['train'].keys())
            random.shuffle(self.train_keys)
        else:
            self.val_keys = list(self.ifile['val'].keys())
        

    def get_batch(self, val = False):
        if not val:
            if len(self.train_keys) < self.n_per:
                self.reset_keys()
                
            keys = ['train/{}'.format(u) for u in self.train_keys[:self.n_per]]
            
            del self.train_keys[:self.n_per]
        else:
            keys = ['val/{}'.format(u) for u in self.val_keys[:self.n_per]]

            del self.val_keys[:self.n_per]
            
        X = []
        Y = []
        edge_index = []
        xg = []
        masks = []
        batch = []
        
        start_node = 0
        counter = 0
        for key in keys:
            x = np.array(self.ifile[key]['x_0']).astype(np.float32)
            y = np.array(self.ifile[key]['y']).astype(np.float32)
            
            x[x == 255] = -1.
            y[y == 255] = -1.
            
            edge_index_ = np.array(self.ifile[key]['edge_index'], dtype = np.int32)
            xg_ = np.array(self.ifile[key]['xg'])
            
            masks.extend(list(np.array(self.ifile[key]['mask'])))
            X.extend(list(x))
            Y.extend(list(y))
            xg.extend(list(xg_))
            
            for k in range(x.shape[0]):
                e_ = edge_index_[k]
                start_node += x.shape[2]
                
                edge_index.append(torch.LongTensor(e_))
                batch.extend(np.ones(x.shape[2], dtype = np.int32) * counter)
                counter += 1
            
        # label smooth
        Y = np.concatenate(Y)
        ey = np.random.uniform(0., 0.1, Y.shape)
        
        Y = Y * (1 - ey) + 0.5 * ey
        
        return torch.FloatTensor(np.concatenate(X)), torch.FloatTensor(Y), edge_index, torch.FloatTensor(np.concatenate(xg)), torch.BoolTensor(np.concatenate(masks) == 1)

class DGLDataGenerator(object):
    def __init__(self, idir, n_sites = 128, 
                 batch_size = 4, val_prop = 0.05, k = 3, pop_size = 300, f_factor = 2):
        self.ifiles = [h5py.File(os.path.join(idir, u), 'r') for u in os.listdir(idir) if u.split('.')[-1] == 'hdf5']
        
        self.training = []
        for ix in range(len(self.ifiles)):
            keys_ = list(self.ifiles[ix].keys())
            
            self.training.extend([(ix, u) for u in keys_])
            
        print(len(self.training))
        n_val = int(len(self.training) * val_prop)
        random.shuffle(self.training)
        
        self.val = self.training[:n_val]
        del self.training[:n_val]
        
        self.n_sites = n_sites
        self.batch_size = batch_size
        
        self.length = len(self.training) // self.batch_size
        self.val_length = len(self.val) // self.batch_size
        
        self.on_epoch_end()
    
        
    def get_element(self, val = False):
        if val:
            if self.val_ix >= len(self.val):
                return None, None, None, None, None
            
            ix, key = self.val[self.val_ix]
            self.val_ix += 1
        else:
            if self.ix >= len(self.training):
                return None, None, None, None, None
            
            ix, key = self.training[self.ix]
            self.ix += 1
            
        bp = np.array(self.ifiles[ix][key]['break_points'])
        if len(bp) <= 1:
            return self.get_element(val)
        
        bp_ix = np.random.choice(range(len(bp) - 1))
        
        bp0 = bp[bp_ix]
        bp1 = bp[bp_ix + 1]
        
        if bp1 - bp0 < self.n_sites:
        
            x = np.array(self.ifiles[ix][key]['x'])[:,bp0:bp1]
            y = np.array(self.ifiles[ix][key]['y'])[:,bp0:bp1]
        else:
            x = np.array(self.ifiles[ix][key]['x'])[:,bp0:bp0 + self.n_sites]
            y = np.array(self.ifiles[ix][key]['y'])[:,bp0:bp0 + self.n_sites]
        
        x = np.pad(x, ((0, 0), (0, self.n_sites - x.shape[1])), constant_values = -1)
        x = np.pad(x, ((0, 299), (0, 0)), constant_values = 0)
        
        y_mask = np.ones(y.shape)
        dpad =  self.n_sites - y.shape[1]
        
        y = np.pad(y, ((0, 0), (0, self.n_sites - y.shape[1])), constant_values = -1)
        y = np.pad(y, ((0, 299), (0, 0)), constant_values = 0)
        
        y_mask = np.pad(y_mask, ((0, 0), (0, dpad)), constant_values = 0)
        y_mask = np.pad(y_mask, ((0, 299), (0, 0)), constant_values = 0)
        y_mask[:150,:] = 0
        
        edges = np.array(self.ifiles[ix][key]['graph']['{}'.format(bp_ix)]['edge_index'])[:,:-1]
        xg = np.array(self.ifiles[ix][key]['graph']['{}'.format(bp_ix)]['xg'])[:-1,:]
        n_mutations = np.array(self.ifiles[ix][key]['graph']['{}'.format(bp_ix)]['n_mutations'])[:-1]
        n_mutations = n_mutations.reshape(n_mutations.shape[0], 1)
        
        edge_attr = np.concatenate([xg[edges[0,:]] - xg[edges[1,:]], n_mutations], axis = 1)
        
        return torch.FloatTensor(x).view(x.shape[0] * x.shape[1], -1), torch.FloatTensor(y).view(x.shape[0] * x.shape[1], -1), edges.reshape(2, edges.shape[0] * edges.shape[2]), torch.FloatTensor(xg).view(x.shape[0] * x.shape[1], -1), torch.BoolTensor(y_mask == 1).view(x.shape[0] * x.shape[1], -1)
                                       
    
    def get_batch(self, val = False):
        xs = []
        ys = []
        edges = []
        masks = []
        batch = []
        xgs = []
        
        current_node = 0
        for ix in range(self.batch_size):
            x, y, e, xg, ym = self.get_element(val)
            
            n = x.shape[0]
            
            edges.append(e)
                
            batch.extend(np.repeat(ix, n))
    
            xs.append(x)
            ys.append(y)
            xgs.append(xg)
            
            current_node += n
            
            masks.append(ym)
            
        x = torch.cat(xs)
        y = torch.cat(ys)
        masks = torch.cat(masks)
        xgs = torch.cat(xgs)

        return x, y, edges, xgs, masks
        
    def on_epoch_end(self):
        random.shuffle(self.training)
        
        self.ix = 0
        self.val_ix = 0
        
        
        
        
        


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
    
# Fibonacci sequence
def F(n):
    if n == 0: return 0
    elif n == 1: return 1
    else: return F(n-1)+F(n-2)
    
EDGE_MEAN = np.array([[1.5445040e+00, 3.9543316e-01, 3.3245298e+03, 4.7975891e+01, 3.3634303e+00,
                      2.6777378e-01, 6.0734084e+07, 1.6249618e+03]])
EDGE_STD = np.array([[2.8873081e+00, 8.6436802e-01, 8.2655654e+03, 9.9553024e+01, 6.8482680e+00,
                     6.8239963e-01, 1.9085554e+08, 7.9583955e+03]])

class GCNDataGeneratorTv2(object):
    def __init__(self, idir, n_sites = 128, 
                 batch_size = 4, val_prop = 0.05, k = 3, pop_size = 300, f_factor = 2):
        self.ifiles = [h5py.File(os.path.join(idir, u), 'r') for u in os.listdir(idir)]
        
        self.training = []
        for ix in range(len(self.ifiles)):
            keys_ = list(self.ifiles[ix])
            
            self.training.extend([(ix, u) for u in keys_])
            
        n_val = int(len(self.training) * val_prop)
        random.shuffle(self.training)
        
        self.val = self.training[:n_val]
        del self.training[:n_val]
        
        self.n_sites = n_sites
        self.batch_size = batch_size
        
        self.length = len(self.training) // self.batch_size
        self.val_length = len(self.val) // self.batch_size
        
        # 1 dilation
        self.nn_samp = list(range(k))
        
        """
        # knn + fibonacci spaced sampling w.r. to topological distance
        ii = 1
        f = F(ii) * f_factor + k - 1
        while f < pop_size - 1:
            self.nn_samp.append(f)
            
            ii += 1
            f = F(ii) * f_factor + k - 1
        """
        self.on_epoch_end()
        self.label_smoothing_factor = 0.1
        
    def on_epoch_end(self):
        random.shuffle(self.training)
        
        self.ix = 0
        self.val_ix = 0
        
    def get_element(self, val = False):
        if val:
            if self.val_ix >= len(self.val):
                return None, None, None
            
            ix, key = self.val[self.val_ix]
            self.val_ix += 1
        else:
            if self.ix >= len(self.training):
                return None, None, None
            
            ix, key = self.training[self.ix]
            self.ix += 1
            
        x = np.array(self.ifiles[ix][key]['x'])
        y = np.array(self.ifiles[ix][key]['y'])
        
        # label smooth
        y_e = np.random.uniform(0., self.label_smoothing_factor, y.shape)
        y = y * (1 - y_e) + 0.5 * y_e
        
        bp = np.array(self.ifiles[ix][key]['break_points'])
        
        if len(bp) == 0:
            return self.get_element(val)
        
        if np.max(bp) <= self.n_sites:
            return self.get_element(val)
        
        s = np.random.choice(range(np.max(bp) - self.n_sites))
        s = list(range(s, s + self.n_sites))
        
        gix = list(np.where((bp >= s[0]) & (bp < s[-1]))[0])
        
        if len(gix) == 0:
            return self.get_element(val)
        
        D = np.zeros((4, 300, 300))
        count = 0
        
        # currently a weighted average of the distance matrices in the region
        for k in gix:
            D_ = np.array(self.ifiles[ix][key]['graph']['{}'.format(k)]['D'])
            
            D_ = np.array([squareform(u) for u in D_], dtype = np.float32)
            
            if bp[k + 1] <= s[-1]:
                w = (bp[k + 1] - bp[k]) / self.n_sites
            else:
                w = (s[-1] - bp[k]) / self.n_sites
        
            D += D_ * w
            count += 1
        
        D /= count
        
        # hop, bran
        # preserve n_mutations as the un-scaled one
        D[1,:,:] *= count
        
        x = x[:,s]
        y = y[150:,s]
        
        bp = [u for u in bp if u in s]        
        bp = list(np.array(bp) - int(np.min(bp)))
        
        edge_index = []
        edge_attr = []
        
        # from pop 1
        for ix in range(D.shape[1] // 2):
            # to pop 1
            ij1 = np.argsort(D[0, ix, :D.shape[1] // 2])[self.nn_samp]
            ij2 = np.argsort(D[0, ix, D.shape[1] // 2:])[self.nn_samp]
            
            # pop 1 -> 2
            edge_class = np.zeros((len(self.nn_samp), 4))
            edge_class[:,2] = 1
            
            # pop 1 -> 1
            edge_class_ = np.zeros((len(self.nn_samp), 4))
            edge_class_[:,0] = 1
            
            edge_class = np.concatenate([edge_class, edge_class_], axis = 0)
            
            ij = np.array(list(ij1) + list(ij2))
            
            _ = np.vstack([(D[:,ix,u] - EDGE_MEAN) / EDGE_STD for u in ij])
        
            ## i -> j
            edge_index.extend([(ix, u) for u in ij])
            edge_attr.extend(list(np.concatenate([edge_class, _], axis = 1)))
           
            
        # from pop 2
        for ix in range(D.shape[0] // 2, D.shape[1]):
            # to pop 1
            ij1 = np.argsort(D[0, ix, :D.shape[1] // 2])[self.nn_samp]
            # to pop 2
            ij2 = np.argsort(D[0, ix, D.shape[1] // 2:])[self.nn_samp]
            
            # pop 2 -> 2
            edge_class = np.zeros((len(self.nn_samp), 4))
            edge_class[:,3] = 1
            
            # pop 2 -> 1
            edge_class_ = np.zeros((len(self.nn_samp), 4))
            edge_class_[:,1] = 1
            
            edge_class = np.concatenate([edge_class, edge_class_], axis = 0)
            
            ij = np.array(list(ij1) + list(ij2))
            
            _ = np.vstack([(D[:,u,ix] - EDGE_MEAN) / EDGE_STD for u in ij])
            
            ## i -> j
            edge_index.extend([(u, ix) for u in ij])
            edge_attr.extend(list(np.concatenate([edge_class, _], axis = 1)))
        
        edge_index = np.array(edge_index, dtype = np.int32)
        edge_attr = np.array(edge_attr, dtype = np.float32)
        
        #np.savez('edge_attrs/edge_attr_{0:04d}.npz'.format(self.ix), edge_attr = edge_attr)
        
        bp_x = np.zeros(x.shape)
        bp_x[:, bp] = 1.
        
        x = np.array((x, bp_x))
        
        #print(edge_attr.shape)
        
        return x, y, torch.LongTensor(edge_index).T, torch.FloatTensor(edge_attr)
    
    def get_batch(self, val = False):
        xs = []
        ys = []
        edges = []
        edge_attr = []
        batch = []
        
        current_node = 0
        for ix in range(self.batch_size):
            x, y, e, ea = self.get_element(val)
    
            if x is None:
                break
            
            n = x.shape[1]
            
            if len(edges) == 0:
                edges = e
            else:
                edges = torch.cat([edges, e + current_node], dim = 1)
                
            batch.extend(np.repeat(ix, n))
    
            xs.append(torch.FloatTensor(x))
            ys.append(torch.FloatTensor(y))
            
            edge_attr.append(ea)
            
            current_node += n
            
        x = torch.stack(xs)
        y = torch.stack(ys)
        edge_attr = torch.cat(edge_attr, dim = 0)
        
        return x, y, edges, edge_attr, torch.LongTensor(batch)
        
class GCNDataGeneratorH5(object):
    def __init__(self, ifile, batch_size = 16, chunk_size = 4):
        self.ifile = h5py.File(ifile, 'r')
        
        self.reset_keys()
        self.reset_keys(True)
        
        self.n_per = batch_size // chunk_size
        
        self.val_length = len(self.val_keys) // self.n_per
        
        
        
    def reset_keys(self, val = False):
        if not val:
            self.train_keys = list(self.ifile['train'].keys())
            random.shuffle(self.train_keys)
        else:
            self.val_keys = list(self.ifile['val'].keys())
        

    def get_batch(self, val = False):
        if not val:
            if len(self.train_keys) < self.n_per:
                self.reset_keys()
                
            keys = ['train/{}'.format(u) for u in self.train_keys[:self.n_per]]
            
            del self.train_keys[:self.n_per]
        else:
            keys = ['val/{}'.format(u) for u in self.val_keys[:self.n_per]]

            del self.val_keys[:self.n_per]
            
        X = []
        Y = []
        edge_index = []
        edge_attr = []
        batch = []
        
        start_node = 0
        counter = 0
        for key in keys:
            x = np.expand_dims(np.array(self.ifile[key]['x_0']), 1)
            
            y = np.array(self.ifile[key]['y'])
            edge_index_ = np.array(self.ifile[key]['edge_index'], dtype = np.int32)
            edge_attr_ = np.array(self.ifile[key]['edge_attr'])
            
            for k in range(edge_attr_.shape[0]):
                edge_attr_[k,:,4:] = (edge_attr_[k,:,4:] - EDGE_MEAN) / EDGE_STD
            
            X.append(x)
            Y.append(y)
            
            for k in range(x.shape[0]):
                e_ = edge_index_[k] + start_node
                start_node += x.shape[2]
                
                edge_index.append(e_)
                batch.extend(np.ones(x.shape[2], dtype = np.int32) * counter)
                counter += 1
                
            edge_attr.extend(list(edge_attr_))
            
        Y = np.concatenate(Y)
        # label smooth
        if not val:
            ey = np.random.uniform(0., 0.1, Y.shape)
            
            Y = Y * (1 - ey) + 0.5 * ey
            
        return torch.FloatTensor(np.concatenate(X)), torch.FloatTensor(Y), torch.LongTensor(np.concatenate(edge_index).T), torch.FloatTensor(np.concatenate(edge_attr)), torch.LongTensor(batch)

class GCNRegDataGeneratorH5(object):
    def __init__(self, ifile, batch_size = 16, chunk_size = 4):
        self.ifile = h5py.File(ifile, 'r')
        
        self.reset_keys()
        self.reset_keys(True)
        
        self.n_per = batch_size // chunk_size
        
        self.val_length = len(self.val_keys) // self.n_per
        
    def reset_keys(self, val = False):
        if not val:
            self.train_keys = list(self.ifile['train'].keys())
            random.shuffle(self.train_keys)
        else:
            self.val_keys = list(self.ifile['val'].keys())
        

    def get_batch(self, val = False):
        if not val:
            if len(self.train_keys) < self.n_per:
                self.reset_keys()
                
            keys = ['train/{}'.format(u) for u in self.train_keys[:self.n_per]]
            
            del self.train_keys[:self.n_per]
        else:
            keys = ['val/{}'.format(u) for u in self.val_keys[:self.n_per]]

            del self.val_keys[:self.n_per]
            
        X = []
        Y = []
        edge_index = []
        edge_attr = []
        batch = []
        
        start_node = 0
        counter = 0
        for key in keys:
            x = np.expand_dims(np.array(self.ifile[key]['x_0']), 1)
            
            y = np.array(self.ifile[key]['y'])
            edge_index_ = np.array(self.ifile[key]['edge_index'], dtype = np.int32)
            edge_attr_ = np.array(self.ifile[key]['edge_attr'])
            
            X.append(x)
            Y.append(y)
            
            for k in range(x.shape[0]):
                e_ = edge_index_[k] + start_node
                start_node += x.shape[2]
                
                edge_index.append(e_)
                batch.extend(np.ones(x.shape[2], dtype = np.int32) * counter)
                counter += 1
                
            edge_attr.extend(list(edge_attr_))
            
        # label smooth
        if not val:
            Y = np.concatenate(Y)
            ey = np.random.uniform(0., 0.1, Y.shape)
            
            Y = Y * (1 - ey) + 0.5 * ey
            
        Y = np.mean(Y, axis = -1).reshape(-1, 1)
            
        return torch.FloatTensor(np.concatenate(X)), torch.FloatTensor(Y), torch.LongTensor(np.concatenate(edge_index).T), torch.FloatTensor(np.concatenate(edge_attr)), torch.LongTensor(batch)

class GCNDataGeneratorT(object):
    def __init__(self, idir, indices, batch_size = 8, 
                     val_prop = 0.05, k = 12, 
                     seg = False, n = 300):
       
        
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
            
        co = []
        for ix in range(150):
            co.append((0, ix))
            
        for ix in range(150, 300):
            co.append((1, ix))
            
        co = np.array(co, dtype = np.float32)
        
        A = kneighbors_graph(co, k, mode='connectivity', include_self = False).tocoo()
        self.edge_index = torch.LongTensor(np.array([A.row, A.col]))
        
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
        
        y = torch.FloatTensor(ifile['y'])[:,:255]
        x = torch.FloatTensor(ifile['x'])[:,:255]
        
        return x, y, self.edge_index
        
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
                edges = torch.cat([edges, e + current_node], dim = 1)
                
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
    def __init__(self, ifile, keys = None, 
                 val_prop = 0.05, batch_size = 16, 
                 chunk_size = 4, pred_pop = 1):
        if keys is None:
            self.keys = list(ifile.keys())
            
            n_val = int(len(self.keys) * val_prop)
            random.shuffle(self.keys)
            
            self.val_keys = self.keys[:n_val]
            del self.keys[:n_val]
            
        self.ifile = ifile
        
        self.chunk_size = chunk_size
        self.batch_size = batch_size
            
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
        
        # label smooth
        ey = np.random.uniform(0, 0.1, Y.shape)
        
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
    
class H5DisDataGenerator_i3(object):
    def __init__(self, ifiles, n_samples = 12000, n_samples_val = 200, chunk_size = 4, batch_size = 32):
        # ifiles is a dictionary pointing to the file names where h5 files corresponding to the class are
        # how many classes / keys?
        self.n_classes = len(ifiles.keys())
        self.classes = sorted(ifiles.keys())
        
        # make a dictionary of the file objects to read
        self.ifiles = dict(zip(self.classes, [h5py.File(ifiles[u][0], 'r') for u in self.classes]))
        
        print(self.ifiles)
        
        self.train_keys = dict(zip(self.classes, [list(self.ifiles[u]['train'].keys()) for u in self.classes]))
        self.val_keys = dict(zip(self.classes, [list(self.ifiles[u]['val'].keys()) for u in self.classes]))
        
        print([len(self.train_keys[u]) for u in self.classes])
        print([len(self.val_keys[u]) for u in self.classes])
        
        self.n_per_class = (batch_size // chunk_size) // self.n_classes
        print(self.n_per_class)
        
        
        self.length = min([len(self.train_keys[u]) // self.n_per_class for u in self.classes])
        self.val_length = min([len(self.val_keys[u]) // self.n_per_class for u in self.classes])
        
        print(self.length, self.val_length)
        
        self.on_epoch_end()
        
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
            else:
                keys = self.val_keys[c][self.ix_val*self.n_per_class : (self.ix_val + 1)*self.n_per_class]
            
            for u in keys:
                if not val:
                    x = np.array(self.ifiles[c]['train'][u]['x_0'], dtype = np.float32)
                else:
                    x = np.array(self.ifiles[c]['val'][u]['x_0'], dtype = np.float32)
                
                Y.extend([self.classes.index(c) for j in range(x.shape[0])])
                X.append(x)
                
                
        if not val:
            self.ix += 1
        else:
            self.ix_val += 1
            
        if len(X) == 0:
            return None, None
        
        X = np.vstack(X)
        
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
            
            for u in keys:
                if not val:
                    x1 = np.array(self.ifiles[c]['train'][u]['x2'], dtype = np.float32)
                    x2 = np.array(self.ifiles[c]['train'][u]['x1'], dtype = np.float32)
                else:
                    x1 = np.array(self.ifiles[c]['val'][u]['x2'], dtype = np.float32)
                    x2 = np.array(self.ifiles[c]['val'][u]['x1'], dtype = np.float32)
                
                Y.extend([self.classes.index(c) for j in range(x1.shape[0])])
                X1.append(np.expand_dims(x1, axis = 1))
                X2.append(np.expand_dims(x2, axis = 1))
                
        if len(X1) == 0:
            return None, None, None
        
        X1 = np.vstack(X1)
        X2 = np.vstack(X2)
        
        return torch.FloatTensor(X1), torch.FloatTensor(X2), torch.LongTensor(Y)
        
class H5DisDataGenerator_i2(object):
    def __init__(self, ifiles, n_samples = 12000, n_samples_val = 200, chunk_size = 4, batch_size = 32):
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
    
    
    