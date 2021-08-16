# -*- coding: utf-8 -*-
import os
import argparse
import logging

import torch
from torch.nn import CrossEntropyLoss, KLDivLoss, NLLLoss, SmoothL1Loss, BCELoss, BCEWithLogitsLoss
import torch

from torch import nn

import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.utils.data
import torch.utils.data.distributed
import torch.nn.functional as F
import torch.distributed as dist
import copy

from layers import PermInvariantClassifier
from data_loaders import H5DisDataGenerator
import h5py

import numpy as np
from sklearn.metrics import accuracy_score
import pandas as pd

bounds = dict()
bounds[0] = (20., 150.) # theta
bounds[1] = (0.01, 0.25) # theta_rho
bounds[2] = (5., 45.) # nu_ab
bounds[3] = (0.01, 3.0) # nu_ba
bounds[4] = (0.01, None) # migTime
bounds[5] = (0.01, 0.7) # migProb
bounds[6] = (0.3, 0.8) # T
bounds[7] = (15000., 80.)

# use this format to tell the parsers
# where to insert certain parts of the script
# ${imports}

SIZE_A = 20
SIZE_B = 14
N_SITES = 10000

import sys
from src.data.data_functions import writeTbsFile, load_data_dros

def normalize(p):
    p[0] = (p[0] - bounds[0][0]) / (bounds[0][1] - bounds[0][0])
    p[1] = (p[1] - bounds[7][0]) / (bounds[7][1] - bounds[7][0])
    p[2] = (p[2] - bounds[2][0]) / (bounds[2][1] - bounds[2][0])
    p[3] = (p[3] - bounds[3][0]) / (bounds[3][1] - bounds[3][0])
    
    p[5] = (p[5] - bounds[5][0]) / (bounds[5][1] - bounds[5][0])
    p[6] = (p[6] - bounds[6][0]) / (bounds[6][1] - bounds[6][0])
    
    return p

def simulate(x):
    
    theta = (bounds[0][1] - bounds[0][0]) * x[0] + bounds[0][0]
    theta_rho = (bounds[1][1] - bounds[1][0]) * x[1] + bounds[1][0]
    
    rho = theta / theta_rho
    nu_a = (bounds[2][1] - bounds[2][0]) * x[2] + bounds[2][0]
    nu_b = (bounds[3][1] - bounds[3][0]) * x[3] + bounds[3][0]
    T = (bounds[6][1] - bounds[6][0]) * x[6] + bounds[6][0]

    b_ = list(bounds[4])
    b_[1] = T / 4.
    migTime = (b_[1] - b_[0]) * x[4] + b_[0]
    migProb = (bounds[5][1] - bounds[5][0]) * x[5] + bounds[5][0]

    p = np.array([theta, rho, nu_a, nu_b, 0, 0, T, T, migTime, 1 - migProb, migTime], dtype = object)

    return p

def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser()
    # my args
    parser.add_argument("--verbose", action = "store_true", help = "display messages")
    parser.add_argument("--weights", default = "None")
    
    parser.add_argument("--ifile", default = "None", help = "initial random simulation data the discriminator was trained on")
    parser.add_arugment("--idir", default = "None")
    
    parser.add_argument("--n_iterations", default = "100")
    
    parser.add_argument("--odir", default = "None")
    parser.add_argument("--T", default = "1.0")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
        logging.debug("running in verbose mode")
    else:
        logging.basicConfig(level=logging.INFO)

    if args.odir != "None":
        if not os.path.exists(args.odir):
            os.system('mkdir -p {}'.format(args.odir))
            logging.debug('root: made output directory {0}'.format(args.odir))
    # ${odir_del_block}

    return args


def main():
    args = parse_args()

    # ${code_blocks}
    device_strings = ['cuda:{}'.format(u) for u in args.devices.split(',')]
    device = torch.device(device_strings[0])

    model = PermInvariantClassifier()
    if len(device_strings) > 1:
        model = nn.DataParallel(model, device_ids = list(map(int, args.devices.split(',')))).to(device)
        model = model.to(device)
    else:
        model = model.to(device)
    
    checkpoint = torch.load(args.weights, map_location = device)
    model.load_state_dict(checkpoint)
    
    model.eval()
    
    ifile = h5py.File(args.ifile, 'r')
    
    # go through the validation keys to get the first initial proposal
    val_keys = list(ifile['val'].keys())
    
    P = []
    y = []
    
    for key in val_keys:
        x1 = torch.FloatTensor(np.expand_dims(np.array(ifile['val'][key]['x1']), axis = 1))
        x2 = torch.FloatTensor(np.expand_dims(np.array(ifile['val'][key]['x2']), axis = 1))
        
        # theta, theta_rho, nu_ab, ...
        p = np.array(ifile['val'][key]['p'])[:,[0, 1, 2, 3, 6, 8, 9]]
        p[:,1] = (p[:,1] ** -1) * p[:,0]
        
        with torch.no_grad():
            y_pred = model(x1, x2).detach().cpu().numpy()
            
            # probability of real classificiation
            y_ = np.exp(y_pred[:,1])
            
        y.extend(list(y_))
        P.extend(list(p))
        
    theta = P[np.argmax(y)].reshape(1, 7)
    theta = normalize(theta)
    
    T = float(args.T)
    accepted = False
    
    odir = os.path.join(args.odir, 'sims')
    os.system('mkdir -p {}'.format(odir))
        
    while not accepted:
        theta += np.random.normal(theta, 1./12. * T, 7)
        x = simulate(theta)
        
        writeTbsFile(x, os.path.join(odir, 'mig.tbs'))

        cmd = "cd %s; %s %d %d -t tbs -r tbs %d -I 2 %d %d -n 1 tbs -n 2 tbs -eg 0 1 6.576808 -eg 0 2 -7.841388 -ma x tbs tbs x -ej tbs 2 1 -en tbs 1 1 -es tbs 2 tbs -ej tbs 3 1 < %s" % (odir, os.path.join(os.getcwd(), 'msmodified/ms'), SIZE_A + SIZE_B, len(P), N_SITES, SIZE_A, SIZE_B, 'mig.tbs')
        print('simulating via the recommended parameters...')
        sys.stdout.flush()
        
        os.system(cmd)
        
        ms = os.path.join(odir, 'mig.msOut')
        anc = os.path.join(odir, 'out.anc')
        
        x1, x2, y1, y2, params = load_data_dros(ms, anc)
        
        x1 = torch.FloatTensor(np.expand_dims(np.array(ifile['val'][key]['x1']), axis = 1))
        x2 = torch.FloatTensor(np.expand_dims(np.array(ifile['val'][key]['x2']), axis = 1))
        
        y_pred = model(x1, x2).detach().cpu().numpy()
        y_ = np.exp(y_pred[:,1])
        
        p = 
        
        
            
            
    
