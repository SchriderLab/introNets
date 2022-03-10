import os
import numpy as np

import sys

from seriate import seriate
import gzip
from scipy.spatial.distance import pdist


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

def load_data_slim(msfile, introgressfile, nindv):
    ig = list(get_gz_file(introgressfile))
    igD = {}
    for x in ig:
        if x[0] == 'Begin':
            n = int(x[-1])
            igD[n] = {}
        if x[0] == 'genome':
            if len(x) > 2:
                igD[n][int(x[1].replace(":", ""))] = [tuple(map(int,i.split('-'))) for i in x[-1].split(',')]
            else:  igD[n][int(x[1].replace(":", ""))] = []           #print(n,x)
    #pprint.pprint(igD)
    g = list(get_gz_file(msfile))
    loc_len = 10000.
    #print(loc_len)
    k = [idx for idx,i in enumerate(g) if len(i) > 0 and i[0].startswith('//')]
    #print(k)
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
        f.append(np.array(q))
        pos_int = np.array(p, dtype='float')

        pos.append(pos_int)

        mask_mat = []
        breakD = igD[gdx]
        for indv in range(len(breakD)):
            mask = binary_digitizer(pos_int, breakD[indv])
            mask_mat.append(mask)

        target.append(np.array(mask_mat, dtype='int8'))
    
    return f, pos, target, igD

def seriate_x(x):
    Dx = pdist(x, metric = 'cosine')
    Dx[np.where(np.isnan(Dx))] = 0.
    ix = seriate(Dx)

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
def load_data(msFile, ancFile):
    msFile = open(msFile, 'r')

    # no migration case
    try:
        ancFile = open(ancFile, 'r')
    except:
        ancFile = None

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
    P = []
    for chunk in ms_chunks:
        pos = np.array([u for u in chunk[2].split(' ')[1:-1] if u != ''], dtype = np.float32)
        
        x = np.array([list(map(int, split(u.replace('\n', '')))) for u in chunk[3:-1]], dtype = np.uint8)
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
            
        X.append(x)
        Y.append(y)
        P.append(pos)
        
    return X, Y, P

def load_data_dros(msFile, ancFile, n_sites = 256, up_sample = False, up_sample_pop_size = 32, filter_zeros = False):
    params = np.loadtxt(os.path.join(os.path.realpath(msFile).replace(msFile.split('/')[-1], ''), 'mig.tbs'), delimiter = ' ')
    msFile = open(msFile, 'r')

    # no migration case
    try:
        ancFile = open(ancFile, 'r')
    except:
        ancFile = None

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
        
if __name__ == '__main__':
    idir = sys.argv[1]
    
    ms_file = os.path.join(idir, 'mig.msOut')
    anc_file = os.path.join(idir, 'anc.out')
    
    x1, x2, y1, y2, params = load_data_dros(ms_file, anc_file)
    
    print(x1[0].shape)
    print(x2[0].shape)
    print(len(x1), len(x2), len(y1), len(y2))
    

