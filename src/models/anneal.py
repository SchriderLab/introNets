# -*- coding: utf-8 -*-
import os
import argparse
import logging

import torch
from torch.nn import CrossEntropyLoss, KLDivLoss, NLLLoss, SmoothL1Loss, BCELoss, BCEWithLogitsLoss
import torch

from torch import nn
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

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
from data_functions import load_npz
import pandas as pd

bounds = dict()
bounds[0] = (1., 150.) # theta
bounds[1] = (0.01, 0.35) # theta_rho
bounds[2] = (1., 200.) # nu_ab
bounds[3] = (0.01, 3.0) # nu_ba
bounds[4] = (0.01, None) # migTime
bounds[5] = (0.01, 1.0) # migProb
bounds[6] = (0.01, 0.9) # T
bounds[7] = (-10., 10.) # alpha1
bounds[8] = (-10, 10.) # alpha2

# use this format to tell the parsers
# where to insert certain parts of the script
# ${imports}

SIZE_A = 20
SIZE_B = 14
N_SITES = 10000

import sys
from data_functions import load_data_dros

import random

def get_real_batch(Xs, batch_size = 64, n_sites = 128):
    x1 = []
    x2 = []
    
    y = []
    for ix in range(batch_size):
        X = Xs[np.random.choice(range(len(Xs)))]
                
        k = np.random.choice(range(X.shape[1] - n_sites))
        
        x1.append(np.expand_dims(X[:20, k : k + n_sites], axis = 0))
        x2.append(np.expand_dims(X[20:, k : k + n_sites], axis = 0))
        
        y.append(1)
        
    return torch.FloatTensor(np.expand_dims(np.concatenate(x1), axis = 1)), torch.FloatTensor(np.expand_dims(np.concatenate(x2), axis = 1)), torch.LongTensor(y)

def writeTbsFile(params, outFileName):
    with open(outFileName, "w") as outFile:
        for paramVec in params:
            outFile.write(" ".join([str(x) for x in paramVec]) + "\n")

def normalize(p):
    rho = p[1]
    theta = p[0]
    
    theta_rho = theta / rho
    
    # theta
    p[0] = (p[0] - bounds[0][0]) / (bounds[0][1] - bounds[0][0])
    p[1] = (theta_rho - bounds[1][0]) / (bounds[1][1] - bounds[1][0])
    
    # nu_a
    p[2] = (p[2] - bounds[2][0]) / (bounds[2][1] - bounds[2][0])
    # nu_b
    p[3] = (p[3] - bounds[3][0]) / (bounds[3][1] - bounds[3][0])
    
    # alpha1
    p[4] = (p[4] - bounds[7][0]) / (bounds[7][1] - bounds[7][0])
    # alpha2
    p[5] = (p[5] - bounds[8][0]) / (bounds[8][1] - bounds[8][0])
    
    # migProb
    p[8] = ((1 - p[8]) - bounds[5][0]) / (bounds[5][1] - bounds[5][0])
    
    # T
    p[6] = (p[6] - bounds[6][0]) / (bounds[6][1] - bounds[6][0])

    
    return p

def simulate(x, n):    
    theta = (bounds[0][1] - bounds[0][0]) * x[0] + bounds[0][0]
    theta_rho = (bounds[1][1] - bounds[1][0]) * x[1] + bounds[1][0]
    
    rho = theta / theta_rho
    nu_a = (bounds[2][1] - bounds[2][0]) * x[2] + bounds[2][0]
    nu_b = (bounds[3][1] - bounds[3][0]) * x[3] + bounds[3][0]
    

    T = (bounds[6][1] - bounds[6][0]) * x[6] + bounds[6][0]
    
    alpha1 = (bounds[7][1] - bounds[7][0]) * x[4] + bounds[7][0]
    alpha2 = (bounds[8][1] - bounds[8][0]) * x[5] + bounds[8][0]

    b_ = list(bounds[4])
    b_[1] = T / 4.
    migTime = (b_[1] - b_[0]) * x[7] + b_[0]
    migProb = (bounds[5][1] - bounds[5][0]) * x[8] + bounds[5][0]

    p = np.tile(np.array([theta, rho, nu_a, nu_b, alpha1, alpha2, 0, 0, T, T, migTime, 1 - migProb, migTime]), (n, 1)).astype(object)

    return p

def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser()
    # my args
    parser.add_argument("--verbose", action = "store_true", help = "display messages")
    parser.add_argument("--weights", default = "None")
    
    #parser.add_argument("--ifile", default = "None", help = "initial random simulation data the discriminator was trained on")
    parser.add_argument("--ifile", default = "None")
    parser.add_argument("--idir", default = "None")
    
    parser.add_argument("--n_steps", default = "45")
    
    parser.add_argument("--odir", default = "None")
    parser.add_argument("--N", default = "100")
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

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def main():
    args = parse_args()
    device = torch.device('cuda')

    model = PermInvariantClassifier()
    model = model.to(device)
    
    checkpoint = torch.load(args.weights, map_location = device)
    model.load_state_dict(checkpoint)
    
    npz = np.load(args.ifile)
    Xs = [load_npz(os.path.join(args.idir, u)).astype(np.uint8) for u in sorted(os.listdir(args.idir))]

    theta = npz['P']
    l = npz['l']
    
    theta = theta[np.argsort(l),:][:int(args.N)]
    for k in range(theta.shape[0]):
        theta[k] = normalize(theta[k])
        
    l = l[np.argsort(l)][:int(args.N)]

    T = float(args.T)

    odir = os.path.join(args.odir, 'sims')
    viz_dir = os.path.join(args.odir, 'viz')
    
    os.system('mkdir -p {}'.format(odir))
    os.system('mkdir -p {}'.format(viz_dir))
    
    criterion = NLLLoss()
    
    history = dict()
    history['step'] = []
    history['gen_loss'] = []
    history['disc_loss'] = []
    
    optimizer = optim.Adam(model.parameters(), lr = 0.001)
        
    for ix in range(int(args.n_steps)):
        viz_dir_ = os.path.join(viz_dir, 'step{0:03d}'.format(ix))
        os.system('mkdir -p {}'.format(viz_dir_))
        
        model.eval()
        
        print('on step {}...'.format(ix))
        print('current nll: {}'.format(np.mean(l)))
        
        X1 = []
        X2 = []
        
        new_theta = theta + np.random.normal(0, 1./12. * T, size = theta.shape)
        new_theta = np.clip(new_theta, 0, 1)

        for k in range(new_theta.shape[0]):
            accepted = False
            
            while not accepted:
                criterion = nn.NLLLoss(reduction = 'none')
                
                new_theta = theta[k] + np.random.normal(0, 1./12. * T, size = theta.shape[1])
                new_theta = np.clip(new_theta, 0, 1)
                
                print('simulating and prediciting on new proposal {}...'.format(k))
            
                x = simulate(new_theta, 100)
                
                writeTbsFile(x, os.path.join(odir, 'mig.tbs'))
        
                cmd = "cd %s; %s %d %d -t tbs -r tbs %d -I 2 %d %d -n 1 tbs -n 2 tbs -eg 0 1 tbs -eg 0 2 tbs -ma x tbs tbs x -ej tbs 2 1 -en tbs 1 1 -es tbs 2 tbs -ej tbs 3 1 < %s" % (odir, os.path.join(os.getcwd(), 'msmodified/ms'), SIZE_A + SIZE_B, len(x), N_SITES, SIZE_A, SIZE_B, 'mig.tbs')
                sys.stdout.flush()
                
                p = os.popen(cmd)
                lines = p.readlines()
                
                f = open(os.path.join(odir, 'mig.msOut'), 'w')
                for line in lines:
                    f.write(line)
                    
                f.close()
                
                ms = os.path.join(odir, 'mig.msOut')
                anc = os.path.join(odir, 'out.anc')
                
                x1, x2, y1, y2, params = load_data_dros(ms, anc)
                x1 = torch.FloatTensor(np.expand_dims(np.array(x1), axis = 1))
                x2 = torch.FloatTensor(np.expand_dims(np.array(x2), axis = 1))
                
                print(x1.shape, x2.shape)
                
                if len(x1.shape) != 4:
                    continue
                
                # theta, theta_rho, nu_ab, nu_ba, alpha1, alpha2, T, migTime, migProb
                p = params[0,[0, 1, 2, 3, 4, 5, 8, 10, 11]]
                
                ys = []
                losses = []
                for c in chunks(list(range(x1.shape[0])), 100):
                    x1_ = x1[c,::].to(device)
                    x2_ = x2[c,::].to(device)
                    target = torch.LongTensor(np.zeros(x1_.shape[0])).to(device)
                    
                    with torch.no_grad():
                        y_pred = model(x1_, x2_)
                        losses.extend(list(criterion(y_pred, target).detach().cpu().numpy().flatten()))
                        
                        # log probability of real classificiation
                        y_ = -y_pred.detach().cpu().numpy()[:,1]
                        
                        ys.extend(list(y_))
                
                fig, axes = plt.subplots(nrows = 2, sharex = True)
                axes[0].hist(ys, bins = 35)
                axes[1].hist(losses, bins = 35)
                
                plt.savefig(os.path.join(viz_dir_, 'prop_{}_hists.png'.format(k)), dpi = 100)
                plt.close()
                
                if (k + 1) % 10 == 0:
                    print('on prop {}...'.format(k + 1))
        
                print('new -ll: {0}, vs. {1}'.format(np.mean(ys), l[k]))
                if np.mean(ys) < l[k]:
                    accepted = True
                else:
                    p = (l[k] / np.mean(ys)) * T
                    
                    if p > 1:
                        accepted = True
                    else:
                        if np.random.choice([0, 1], p = [1 - p, p]) == 1:
                            accepted = True
                            
                    
            print('accepted a new theta...')
            print(np.mean(losses))
            theta[k] = copy.copy(new_theta)
    
            l[k] = np.mean(ys)
            
            X1.append(x1)
            X2.append(x2)
    
        theta_ = np.concatenate([simulate(theta[k], 1) for k in range(theta.shape[0])])
        np.savez(os.path.join(args.odir, 'theta_{0:4d}.npz'.format(ix)), theta = theta_, l = np.array(l))
        
        history['step'].append(ix)
        history['gen_loss'].append(np.mean(l))
                
        T -= 0.02
        
        X1 = torch.cat(X1)
        X2 = torch.cat(X2)
        
        indices = list(range(X1.shape[0]))
        random.shuffle(indices)
        
        # training phase
        model.train()
        criterion = nn.NLLLoss()
        
        losses = []
        accuracies = []
        
        counter = 0
        print('performing an epoch of training...')
        for c in chunks(indices, 99):
            optimizer.zero_grad()
            
            x1 = X1[c,::].to(device)
            x2 = X2[c,::].to(device)
            y = torch.LongTensor(np.zeros(x1.shape[0])).to(device)
            
            y_pred = model(x1, x2)
            
            x1r, x2r, yr = get_real_batch(Xs)
            x1r = x1r.to(device)
            x2r = x2r.to(device)
            yr = yr.to(device)
            
            y_pred_real = model(x1r, x2r)

            loss = (criterion(y_pred, y) + criterion(y_pred_real, yr)) * 0.5 # ${loss_change}
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

            # compute accuracy in CPU with sklearn
            y_pred = np.exp(y_pred.detach().cpu().numpy())
            y = y.detach().cpu().numpy()

            y_pred = np.argmax(y_pred, axis=1)

            # append metrics for this epoch
            accuracies.append(accuracy_score(y, y_pred))
            
            print(loss.item())
                
            counter += 1
            
        plt.hist(losses, bins = 35)
        plt.savefig(os.path.join(viz_dir_, 'disc_losses.png'), dpi = 100)
        plt.close()
                    
        print('have {0} as the new loss and {1} acc as the loss of the discriminator...'.format(np.mean(losses), np.mean(accuracies)))
        
        history['disc_loss'].append(np.mean(losses))
        
        df = pd.DataFrame(history)
        df.to_csv(os.path.join(args.odir, 'history.csv'), index = False)
        
        print('saving weights...')
        torch.save(model.state_dict(), os.path.join(args.odir, 'disc_{0:04d}.weights'.format(ix)))
        
        
        
            
if __name__ == '__main__':
    main()
    
