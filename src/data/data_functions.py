import os
import numpy as np

import sys

def split(word):
    return [char for char in word]

def load_data_dros(msFile, ancFile, n_sites = 64, up_sample = False, up_sample_pop_size = 32):
    params = np.loadtxt(os.path.join(os.path.realpath(msFile).replace(msFile.split('/')[-1], ''), 'mig.tbs'), delimiter = ' ')
    msFile = open(msFile, 'r')

    # no migration case
    try:
        ancFile = open(ancFile, 'r')
    except:
        ancFile = None

    ms_lines = msFile.readlines()[:-1]

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
        
    X1 = []
    X2 = []
    
    Y1 = []
    Y2 = []

    for chunk in ms_chunks:
        pos = np.array([u for u in chunk[2].split(' ')[1:-1] if u != ''], dtype = np.float32)

        x = np.array([list(map(int, split(u.replace('\n', '')))) for u in chunk[3:-1]], dtype = np.uint8)
        if anc_lines is not None:
            y = np.array([list(map(int, split(u.replace('\n', '')))) for u in anc_lines[:len(pos)]], dtype = np.uint8)
            y = y.T

            del anc_lines[:len(pos)]
        else:
            y = np.zeros(x.shape, dtype = np.uint8)
                    
        k = np.random.choice(range(x.shape[1] - n_sites))
        
        if len(y.shape) > 1:
        
            pop1_x = x[:20, k:k + n_sites]
            pop2_x = x[20:, k:k + n_sites]
    
            pop1_y = y[:20, k:k + n_sites]
            pop2_y = y[20:, k:k + n_sites]
            
            X1.append(pop1_x)
            X2.append(pop2_x)
            
            Y1.append(pop1_y)
            Y2.append(pop2_y)
        
    return X1, X2, Y1, Y2, params
        
if __name__ == '__main__':
    idir = sys.argv[1]
    
    ms_file = os.path.join(idir, 'mig.msOut')
    anc_file = os.path.join(idir, 'anc.out')
    
    x1, x2, y1, y2, params = load_data_dros(ms_file, anc_file)
    
    print(x1[0].shape)
    print(x2[0].shape)
    print(len(x1), len(x2), len(y1), len(y2))
    

