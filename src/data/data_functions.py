import os
import numpy as np

from seriate import seriate
import gzip
from scipy.spatial.distance import pdist
from calc_stats_ms import N_ton, distance_vector, min_dist_ref, s_star_ind, num_private, label

from seriate import seriate
from scipy.spatial.distance import pdist, cdist

from scipy.optimize import linear_sum_assignment
from scipy.interpolate import interp1d
from collections import deque

import logging
import time

def make_continuous(x):
    x = np.cumsum(x, axis = 1) * 2 * np.pi

    mask = np.zeros(x.shape)
    ix = list(np.where(np.diff(x) != 0))
    ix[-1] += 1
    mask[tuple(ix)] = 1
    mask[:,-1] = 1
    
    x[mask == 0] = 0
    t = np.array(range(x.shape[-1]))
    
    for k in range(len(x)):
        ix = [0] + list(np.where(x[k] != 0)[0])
        print(len(ix))
        
        t = np.array(range(np.max(ix)))
        
        if len(ix) > 3:
            x[k,:len(t)] = interp1d(ix, x[k,ix], kind = 'cubic')(t)
        elif len(ix) > 2:
            x[k,:len(t)] = interp1d(ix, x[k,ix], kind = 'quadratic')(t)
        elif len(ix) > 1:
            x[k,:len(t)] = interp1d(ix, x[k,ix], kind = 'linear')(t)
            
    x = np.cos(x)
    return x

def seriate_x(x, metric = 'cosine'):
    Dx = pdist(x, metric = metric)
    Dx[np.where(np.isnan(Dx))] = 0.
    ix = seriate(Dx, timeout = 0)

    return x[ix], ix


#### TwoPopAlignmentFormatter
## Class to take MS or SLiM output of alignments from two-pop simulations and filter and format them.
## this class is to be used after actually reading the files with the I/O functions below.
# expects:
    ## x - (list) of numpy arrays of binary alignments
    ## y - (list or None) of numpy arrays of the segmented alignments 
    ## p - (list or None) of relevant params (elements are array like) we wish to correspond to the formatted data returned
    ## shape - (tuple) desired output shape (channels, individuals, sites) or the predictor X variable.  The last two dimensions will match Y
    ## pop_sizes - (tuple) pop sizes of each population (we expect two here)
    ## sorting - (str) type of sorting to perform during formatting.  The only supported options r.n. are seriate_match or None
    ## pop - (int or None) which population to save in the channels output of the Y variable. [0 or 1]
class TwoPopAlignmentFormatter(object):
    def __init__(self, x, y = None, p = None, shape = (2, 32, 64), pop_sizes = [150, 156], 
                 sorting = 'seriate_match', sorting_metric = 'cosine', pop = None, seriation_pop = 0, unphased = False):
        # list of x and y arrays
        self.x = deque(x)
        if y is not None:
            self.y = deque(y)
        else:
            self.y = y
            
        if p is not None:
            self.p = deque(p)
        else:
            self.p = p
        
        self.seriation_pop = seriation_pop
        self.n_pops = shape[0]
        self.pop_size = shape[1]
        self.n_sites = shape[2]
        
        self.pop_sizes = pop_sizes
        self.sorting = sorting
        self.pop = pop
        self.metric = sorting_metric
        self.unphased = unphased
        
        self.iter = 0
        
    # splits and upsamples the sub-populations if needed
    def resample_split(self, x):
        # upsample the populations if need be
        x1_indices = list(range(self.pop_sizes[0]))
        n = self.pop_size - self.pop_sizes[0]
        
        if n > self.pop_sizes[0]:
            replace = True
        else:
            replace = False
        
        if n > 0:
            x1_indices = x1_indices + list(np.random.choice(range(self.pop_sizes[0]), n, replace = replace))
        
        # upsample the second pop (again if needed)
        x2_indices = list(range(self.pop_sizes[0], self.pop_sizes[0] + self.pop_sizes[1]))
        n = self.pop_size - self.pop_sizes[1]
        
        if n > self.pop_sizes[1]:
            replace = True
        else:
            replace = False
        
        if n > 0:
            x2_indices = x2_indices + list(np.random.choice(range(self.pop_sizes[0], self.pop_sizes[0] + self.pop_sizes[1]), n, replace = replace))
        
        x1 = x[x1_indices, :]
        x2 = x[x2_indices, :]
        
        if self.unphased:
            x1 = x1[:len(x1) // 2] + x1[len(x1) // 2:]
            x2 = x2[:len(x2) // 2] + x2[len(x2) // 2:]
            
            
        
        return x1, x2, x1_indices, x2_indices

    # store a list of the desired arrays
    ## args:
        # include_zeros: (bool) whether to include random samples that have no-introgression (no positive pixels)
    def format(self, include_zeros = False):
        logging.debug('formatting {} random windows...'.format(len(self.x)))
        if self.y is not None:
            assert len(self.x) == len(self.y), 'y and x are not the same length!'
        
        X = []
        Y = []
        P = []
        Indices = []
        
        self.six = []
        
        t_seriation = []
        t_matching = []
        
        while len(self.x) > 0:
            x = self.x.pop()
            logging.debug('have initial shape of {}...'.format(x.shape))
            
            if self.p is not None:
                p = self.p.pop()
            
            if self.y is not None:
                y = self.y.pop()
                
            if x.shape[1] < self.n_sites:
                logging.debug('while formatting, found iter {} to have to few sites to format.  Given MS will return a varying amount of segregating sites per sim, this is expected. \n However if you find this to happen too often try increasing the mutation rate or the size of the simulated chromosomes.'.format(self.iter))
                continue
            
            x1, x2, x1_indices, x2_indices = self.resample_split(x)
            
            logging.debug('have shapes {}, {}'.format(x1.shape, x2.shape))
            
            # in this block we down-sample by randomly selecting a window of the desired size from the simulation replicate
            if self.y is not None:
                y1 = y[x1_indices, :]
                y2 = y[x2_indices, :]
             
                if self.pop == 0:
                    yi = y1
                elif self.pop == 1:
                    yi = y2
                else:
                    yi = np.concatenate([y1, y2], axis = 0)
             
                if not include_zeros:
                    indices = list(set(range(x1.shape[1] - self.n_sites)).intersection(list(np.where(np.sum(yi, axis = 0) > 0)[0])))
                    if len(indices) == 0:
                        logging.debug('no introgression in this sim...skipping it...')
                        continue
                else:
                    indices = list(range(x1.shape[1] - self.n_sites + 1))
                
                six = np.random.choice(indices)
                logging.debug('chose random starting index: {}'.format(six))

                y1 = y1[:,six:six + self.n_sites]
                y2 = y2[:,six:six + self.n_sites]
                
                if y1.shape[1] != self.n_sites:
                    print('didnt find the correct number of sites in y...')
                    continue 
                
                if self.unphased:
                    y1 = np.bitwise_or(y1[:len(y1) // 2], y1[len(y1) // 2:])
                    y2 = np.bitwise_or(y2[:len(y2) // 2], y2[len(y2) // 2:])
                
            else:
                indices = list(range(x1.shape[1] - self.n_sites + 1))
                six = np.random.choice(indices)
                
            x1 = x1[:,six:six + self.n_sites]
            x2 = x2[:,six:six + self.n_sites]
            
            
            
            x_sfs = np.sum(x1, axis = 0) + np.sum(x2, axis = 0)
            
            #### do sorting ------
            if self.sorting == "seriate_match":
                if self.seriation_pop == 0:
                    # time the seriation operation
                    t0 = time.time()
                    
                    x1, ix1 = seriate_x(x1, self.metric)
                    t_ser = time.time() - t0
                    
                    # time the least cost linear matching
                    t0 = time.time()
                    D = cdist(x1, x2, metric = self.metric)
                    D[np.where(np.isnan(D))] = 0.
                    
                    i, j = linear_sum_assignment(D)
                    t_match = time.time() - t0
                    
                    x2 = x2[j,:]
                    
                    x1_indices = [x1_indices[u] for u in ix1]
                    x2_indices = [x2_indices[u] for u in j]
                    
                    if self.y is not None:
                        y1 = y1[ix1, :]
                        y2 = y2[j, :]
                else:
                    # time the seriation operation
                    t0 = time.time()
                    
                    x2, ix2 = seriate_x(x2, self.metric)
                    t_ser = time.time() - t0
                    
                    t0 = time.time()
                    D = cdist(x2, x1, metric = self.metric)
                    D[np.where(np.isnan(D))] = 0.
                    
                    i, j = linear_sum_assignment(D)
                    t_match = time.time() - t0
                    
                    x1 = x1[j,:]
                    
                    x1_indices = [x1_indices[u] for u in j]
                    x2_indices = [x2_indices[u] for u in ix2]
                    
                    if self.y is not None:
                        y1 = y1[j, :]
                        y2 = y2[ix2, :]
                
                t_seriation.append(t_ser)
                t_matching.append(t_match)
            
            x = np.array([x1, x2])
            
            assert (np.sum(x, axis = 0).sum(axis = 0) == x_sfs).all()
            
            self.six.append(six)
            if self.y is not None:
                y = np.array([y1, y2])
            
            
            if self.y is not None:
                ys1 = np.sum(y1, axis = 0)
                ys2 = np.sum(y2, axis = 0)
                assert (np.sum(y[0], axis = 0) == ys1).all() and (np.sum(y[1], axis = 0) == ys2).all()
                
                if self.pop == 0:
                    y = np.expand_dims(y[0], axis = 0)
                elif self.pop == 1:
                    y = np.expand_dims(y[1], axis = 0)
                        
                Y.append(y)
            else:
                Y = None
            
            if self.p is not None:
                P.append(p)
            
            X.append(x)            
            Indices.append((x1_indices, x2_indices))
            
            logging.debug('x1 ix: {}'.format(x1_indices))
            logging.debug('x2 ix: {}'.format(x2_indices))
            
        self.x = X
        self.y = Y
        self.indices = Indices
        self.p = P
        
        self.time = (t_seriation, t_matching)
        
# Numpy version of:
def batch_dot(W, x):
    # dot product for a batch of examples
    return np.einsum("ijk,ki->ji", np.tile(W, (len(x), 1, 1)), x.T).T

def read_slim_out(ifile):
    ifile = open(ifile, 'r')
    
    lines = ifile.readlines()

    migProbs = []
    migTimes = []
    for line in lines:
        line = line.replace('\n', '').replace(',', '')
        if 'migTime' in line:
            migTimes.append(float(line.split(' ')[-1]))
        elif 'migProbs' in line:
            migProbs.append(tuple(map(float, line.split(' ')[-2:])))
            
    return migProbs, migTimes

def get_feature_vector(mutation_positions, genotypes, ref_geno, arch):
    n_samples = len(genotypes[0])

    n_sites = 50000

    ## set up S* stuff -- remove mutations found in reference set
    t_ref = list(map(list, zip(*ref_geno)))
    t_geno = list(map(list, zip(*genotypes)))
    pos_to_remove = set()  # contains indexes to remove
    s_star_haps = []
    for idx, hap in enumerate(t_ref):
        for jdx, site in enumerate(hap):
            if site == 1:
                pos_to_remove.add(jdx)

    for idx, hap in enumerate(t_geno):
        s_star_haps.append([v for i, v in enumerate(hap) if i not in pos_to_remove])

    ret = []

    # individual level
    for focal_idx in range(0, n_samples):
        t0 = time.time()
        calc_N_ton = N_ton(genotypes, n_samples, focal_idx)
        #print('calc_N_ton: {}'.format(time.time() - t0))
        
        t0 = time.time()
        dist = distance_vector(genotypes, focal_idx)
        #print('dist: {}'.format(time.time() - t0))
        
        t0 = time.time()
        min_d = [min_dist_ref(genotypes, ref_geno, focal_idx)]
        #print('min_d: {}'.format(time.time() - t0))
        
        t0 = time.time()
        ss = [s_star_ind(np.array(s_star_haps), np.array(mutation_positions), focal_idx)]
        #print('ss: {}'.format(time.time() - t0))
        
        t0 = time.time()
        n_priv = [num_private(np.array(s_star_haps), focal_idx)]
        #print('n_priv: {}'.format(time.time() - t0))
        
        t0 = time.time()
        focal_arch = [row[focal_idx] for row in arch]
        #print('focal_arch: {}'.format(time.time() - t0))
        
        t0 = time.time()
        lab = label(focal_arch, mutation_positions, n_sites, 0.7, 0.3)
        #print('label: {}'.format(time.time() - t0))

        output = calc_N_ton + dist + min_d + ss + n_priv + lab
        ret.append(output)

    return np.array(ret, dtype = np.float32)

# writes a space separated .tbs file (text)
# which contains the demographic parameters for an ms two-population simulation
def writeTbsFile(params, outFileName):
    with open(outFileName, "w") as outFile:
        for paramVec in params:
            outFile.write(" ".join([str(x) for x in paramVec]) + "\n")

# finds the middle component of a list (if there are an even number of entries, then returns the first of the middle two)
def findMiddle(input_list):
    middle = float(len(input_list))/2
    if middle % 2 != 0:
        return input_list[int(middle - .5)]
    else:
        return (input_list[int(middle)], input_list[int(middle-1)])[0]
    
# return the indices of positions that fall in sliding windows wsize long
# and with a step size = step
def get_windows(ipos, wsize = 50000, step = 10000):
    # positions of polymorphisms
    s1, s2 = 0, wsize
    
    mp = np.max(ipos)

    ret = []
    while s2 <= mp:
        ix = list(np.where((ipos >= s1) & (ipos <= s2))[0])
        
        s1 += step
        s2 += step
        
        ret.append(ix)
        
    ret.append(ix)
    
    return ret

def binary_digitizer(x, breaks):
    #x is all pos of seg sites
    #breaks are the introgression breakpoints, as a list of lists like [[1,4], [22,57], [121,144]....]
    #output is a numpy vector with zero for all pos not in introgression and one for all points in introgression
    flat_breaks = np.array(breaks).flatten()
    lenx = len(x)
    lzero, rzero = np.zeros(lenx), np.zeros(lenx)
    dg_l = np.digitize(x, flat_breaks, right=False)
    dg_r = np.digitize(x, flat_breaks, right=True)
    lzero[dg_l % 2 > 0] = 1
    rzero[dg_r % 2 > 0] = 1
    return np.array([lzero, rzero]).max(axis=0)

def get_gz_file(filename, splitchar = 'NA', buffered = False):
    if not buffered:
        if splitchar == 'NA':
            return [i.strip().split() for i in gzip.open(filename, 'rt')]
        else: return [i.strip().split(splitchar) for i in gzip.open(filename, 'rt')]
    else:
        if splitchar == 'NA':
            return (i.strip().split() for i in gzip.open(filename, 'rt'))
        else: return (i.strip().split(splitchar) for i in gzip.open(filename, 'rt'))

def load_data_slim(msfile, introgressfile, nindv, region = None):
    ig = list(get_gz_file(introgressfile))
    igD = {}
    for x in ig:
        if x[0] == 'Begin':
            n = int(x[-1])
            igD[n] = {}
        if x[0] == 'genome':
            if len(x) > 2:
                igD[n][int(x[1].replace(":", ""))] = [tuple(map(int,i.split('-'))) for i in x[-1].split(',')]
            else:  igD[n][int(x[1].replace(":", ""))] = []           
    
    g = list(get_gz_file(msfile))
    loc_len = 10000.

    k = [idx for idx,i in enumerate(g) if len(i) > 0 and i[0].startswith('//')]

    f, pos, target = [], [], []
    for gdx,i in enumerate(k):
        L = g[i+3:i+3+nindv]
        p = [jj for jj in g[i+2][1:]]
        q = []
        kdx = 1
        for i in L:
            
            i = [int(j) for j in i[0]]

            i = np.array(i, dtype=np.int8)
            q.append(i)

        q = np.array(q)
        q = q.astype("int8")
        
        pos_int = np.array(p, dtype='float')
        
        mask_mat = []
        breakD = igD[gdx]
        for indv in range(len(breakD)):
            mask = binary_digitizer(pos_int, breakD[indv])
            mask_mat.append(mask)

        mask_mat = np.array(mask_mat, dtype='int8')

        f.append(q)
        pos.append(pos_int)
        target.append(mask_mat)
        
    
    
    return f, pos, target

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

def split(word):
    return [char for char in word]

######
# generic function for msmodified
# ----------------
def load_data(msFile, ancFile = None, n = None, leave_out_last = False, region = None):
    msFile = gzip.open(msFile, 'r')

    # no migration case
    if ancFile is not None:
        ancFile = gzip.open(ancFile, 'r')

    ms_lines = [u.decode('utf-8') for u in msFile.readlines()]

    if leave_out_last:
        ms_lines = ms_lines[:-1]

    idx_list = [idx for idx, value in enumerate(ms_lines) if '//' in value] + [len(ms_lines)]
    
    ms_chunks = [ms_lines[idx_list[k]:idx_list[k+1]] for k in range(len(idx_list) - 1)]
    ms_chunks[-1] += ['\n']

    if ancFile is not None:
        anc_lines = [u.decode('utf-8') for u in ancFile.readlines()]
    else:
        anc_lines = None
        
    X = []
    Y = []
    P = []
    for chunk in ms_chunks:
        pos = np.array([u for u in chunk[2].split(' ')[1:-1] if u != ''], dtype = np.float32)

        x = np.array([list(map(int, split(u.replace('\n', '')))) for u in chunk[3:3 + n]], dtype = np.uint8)
        # destroy the perfect information regarding
        # which allele is the ancestral one
        for k in range(x.shape[1]):
            if np.sum(x[:,k]) > x.shape[0] / 2.:
                x[:,k] = 1 - x[:,k]
            elif np.sum(x[:,k]) == x.shape[0] / 2.:
                if np.random.choice([0, 1]) == 0:
                    x[:,k] = 1 - x[:,k]
        
        if anc_lines is not None:
            y = np.array([list(map(int, split(u.replace('\n', '')))) for u in anc_lines[:len(pos)]], dtype = np.uint8)
            y = y.T
            
            del anc_lines[:len(pos)]
        else:
            y = np.zeros(x.shape, dtype = np.uint8)
            
        assert len(pos) == x.shape[1]
            
        if region is not None:
            ii = np.where((pos >= region[0]) & (pos <= region[1]))[0]
            x = x[:,ii]
            y = y[:,ii]
            pos = pos[ii]
            
        X.append(x)
        Y.append(y)
        P.append(pos)
        
    return X, Y, P

def load_data_dros(msFile, ancFile = None, n_sites = None, up_sample = False, up_sample_pop_size = 32, filter_zeros = False):
    params = np.loadtxt(os.path.join(os.path.realpath(msFile).replace(msFile.split('/')[-1], ''), 'mig.tbs'), delimiter = ' ')
    msFile = open(msFile, 'r')

    # no migration case
    if ancFile is not None:
        ancFile = open(ancFile, 'r')

    ms_lines = msFile.readlines()

    if ancFile is not None:
        idx_list = [idx for idx, value in enumerate(ms_lines) if '//' in value] + [len(ms_lines)]
    else:
        idx_list = [idx for idx, value in enumerate(ms_lines) if '//' in value] + [len(ms_lines)]

    ms_chunks = [ms_lines[idx_list[k]:idx_list[k+1]] for k in range(len(idx_list) - 1)]
    ms_chunks[-1] += ['\n']

    if ancFile is not None:
        anc_lines = ancFile.readlines()
    else:
        anc_lines = None
        
    X = []
    Y = []
    for chunk in ms_chunks:
        pos = np.array([u for u in chunk[2].split(' ')[1:-1] if u != ''], dtype = np.float32)

        x = np.array([list(map(int, split(u.replace('\n', '')))) for u in chunk[3:-1]], dtype = np.uint8)
        if anc_lines is not None:
            y = np.array([list(map(int, split(u.replace('\n', '')))) for u in anc_lines[:len(pos)]], dtype = np.uint8)
            y = y.T

            del anc_lines[:len(pos)]
        else:
            y = np.zeros(x.shape, dtype = np.uint8)
                    
        if n_sites is not None:
            n = x.shape[1]
            
            x = np.pad(x, ((0, 0), (0, n_sites - n)))
            y = np.pad(y, ((0, 0), (0, n_sites - n)))
        
        if filter_zeros:
            if np.sum(y) > 0:
                X.append(x)
                Y.append(y)
        else:
            X.append(x)
            Y.append(y)
        
    return X, Y, params
        

    

