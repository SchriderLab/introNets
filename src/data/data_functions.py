import os
import numpy as np

from seriate import seriate
import gzip
from scipy.spatial.distance import pdist
from calc_stats_ms import N_ton, distance_vector, min_dist_ref, s_star_ind, num_private, label

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
            migTimes.append(tuple(map(float, line.split(' ')[-2:])))
            
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
        calc_N_ton = N_ton(genotypes, n_samples, focal_idx)
        dist = distance_vector(genotypes, focal_idx)
        min_d = [min_dist_ref(genotypes, ref_geno, focal_idx)]
        ss = [s_star_ind(np.array(s_star_haps), np.array(mutation_positions), focal_idx)]
        n_priv = [num_private(np.array(s_star_haps), focal_idx)]
        focal_arch = [row[focal_idx] for row in arch]
        lab = label(focal_arch, mutation_positions, n_sites, 0.7, 0.3)

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
def load_data(msFile, ancFile, leave_out_last = False):
    msFile = open(msFile, 'r')

    # no migration case
    try:
        ancFile = open(ancFile, 'r')
    except:
        ancFile = None

    ms_lines = msFile.readlines()

    if leave_out_last:
        ms_lines = ms_lines[:-1]

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
        

    

