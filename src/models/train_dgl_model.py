# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
import os
import argparse
import logging

import torch
from torch.nn import CrossEntropyLoss, KLDivLoss, NLLLoss, SmoothL1Loss, BCELoss, BCEWithLogitsLoss
import torch
torch.autograd.set_detect_anomaly(True)

from torch import nn

import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.utils.data
import torch.utils.data.distributed
import torch.nn.functional as F
import torch.distributed as dist
import copy

import h5py

import numpy as np
from sklearn.metrics import accuracy_score
import pandas as pd
# use this format to tell the parsers
# where to insert certain parts of the script
# ${imports}

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from layers import GCNUNet_i2, GCNUNet_i3
from data_loaders import GCNDataGenerator, DGLDataGenerator
import glob

from scipy.special import expit

from dgl_layers import TreeLSTM
import dgl

def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser()
    # my args
    parser.add_argument("--verbose", action = "store_true", help = "display messages")
    parser.add_argument("--ifile", default = "BA_10e6_seriated.hdf5")
    parser.add_argument("--idir", default = "None")

    parser.add_argument("--weights", default = "None", help = "weights to load (optional)")

    parser.add_argument("--devices", default = "0")
    parser.add_argument("--n_plateau", default = "5")
    parser.add_argument("--rl_factor", default = "0.5")
    parser.add_argument("--n_epochs", default = "100")
    parser.add_argument("--n_early", default = "10")
    parser.add_argument("--indices", default = "None")
    
    parser.add_argument("--n_heads", default = "2")

    parser.add_argument("--batch_size", default = "16")
    
    parser.add_argument("--seg", action = "store_true")
    parser.add_argument("--loss", default = "bce")
    parser.add_argument("--layer_type", default = "gat")
    
    parser.add_argument("--n_features", default = "128")
    parser.add_argument("--n_global", default = "1024")
    
    parser.add_argument("--n_layers", default = "11")
    # ${args}

    parser.add_argument("--odir", default = "training_output")
    parser.add_argument("--ofile", default = "None")
    parser.add_argument("--tag", default = "test")
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
    
    model = TreeLSTM()
    print(model)
    
    
    if len(device_strings) > 1:
        model = nn.DataParallel(model, device_ids = list(map(int, args.devices.split(',')))).to(device)
        model = model.to(device)
    else:
        model = model.to(device)
        
    if args.weights != "None":
        checkpoint = torch.load(args.weights, map_location = device)
        model.load_state_dict(checkpoint)
        
    generator = DGLDataGenerator(args.idir,
                                 batch_size = int(args.batch_size))
    

    criterion = nn.BCEWithLogitsLoss(pos_weight = torch.FloatTensor([0.6713357505900737])).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr = 0.001)
    early_count = 0
    
    decayRate = 0.99
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer = optimizer, gamma=decayRate)
    
    history = dict()
    history['loss'] = []
    history['epoch'] = []
    history['val_loss'] = []

    min_val_loss = np.inf
    print('training...')
    for ix in range(int(args.n_epochs)):
        model.train()

        losses = []
        accuracies = []

        for ij in range(generator.length):
            optimizer.zero_grad()
            
            x, y, edges, xg, y_mask = generator.get_batch()
            if x is None:
                break
            
            batch = dgl.batch([dgl.graph((u[1,:].to(device), u[0,:].to(device))) for u in edges])
            batch.ndata['x'] = x.to(device)
            
            n = x.shape[0]
            
            # include the pop and time of the node in the hidden state
            h = torch.cat([xg, torch.zeros((n, 380))], dim = 1).to(device)
            c = torch.zeros((n, 384)).to(device)
            y = y.to(device)
            y_mask = y_mask.to(device)

            y_pred = model(batch, h, c)
            #print(y.shape, y_pred.shape)

            loss = criterion(torch.masked_select(y_pred, y_mask), torch.masked_select(y, y_mask)) # ${loss_change}

            loss.backward()
            optimizer.step()
            losses.append(loss.item())

            # compute accuracy in CPU with sklearn
            y_pred = np.round(expit(y_pred.detach().cpu().numpy()[y_mask.detach().cpu().numpy()]).flatten())
            y = np.round(y.detach().cpu().numpy()[y_mask.detach().cpu().numpy()].flatten())

            # append metrics for this epoch
            accuracies.append(accuracy_score(y, y_pred))

            if (ij + 1) % 5 == 0:
                logging.info(
                    'root: Epoch {0}, step {3}: got loss of {1}, acc: {2}'.format(ix, np.mean(losses),
                                                                                  np.mean(accuracies), ij + 1))

        model.eval()
        
        history['epoch'].append(ix)
        history['loss'].append(np.mean(losses))

        val_losses = []
        val_accs = []
        
        Y = []
        Y_pred = []
        for step in range(generator.val_length):
            with torch.no_grad():
                x, y, edges, xg, y_mask = generator.get_batch(val = True)
                if x is None:
                    break
                
                batch = dgl.batch([dgl.graph((u[1,:].to(device), u[0,:].to(device))) for u in edges])
                batch.ndata['x'] = x.to(device)
                
                n = x.shape[0]
                
                # include the pop and time of the node in the hidden state
                h = torch.cat([xg, torch.zeros((n, 380))], dim = 1).to(device)
                c = torch.zeros((n, 384)).to(device)
                y = y.to(device)
                y_mask = y_mask.to(device)
    
                y_pred = model(batch, h, c)
                #print(y.shape, y_pred.shape)
    
                loss = criterion(torch.masked_select(y_pred, y_mask), torch.masked_select(y, y_mask)) # ${loss_change}
                val_losses.append(loss.detach().item())
                
                # compute accuracy in CPU with sklearn
                y_pred = np.round(expit(y_pred.detach().cpu().numpy()[y_mask.detach().cpu().numpy()]).flatten())
                y = np.round(y.detach().cpu().numpy()[y_mask.detach().cpu().numpy()].flatten())
                
                val_accs.append(accuracy_score(np.round(y), np.round(y_pred)))

        
        val_loss = np.mean(val_losses)
        history['val_loss'].append(val_loss)

        logging.info(
            'root: Epoch {0}, got val loss of {1}, acc {2}'.format(ix, val_loss, np.mean(val_accs)))
        
        # ${save_extra_history}

        val_loss = np.mean(val_losses)
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            print('saving weights...')
            torch.save(model.state_dict(), os.path.join(args.odir, '{0}.weights'.format(args.tag)))

            early_count = 0
        else:
            early_count += 1

            # early stop criteria
            if early_count > int(args.n_early):
                break

        generator.on_epoch_end()
        
        df = pd.DataFrame(history)
        df.to_csv(os.path.join(args.odir, '{}_history.csv'.format(args.tag)), index = False)
        
        lr_scheduler.step()

if __name__ == '__main__':
    main()
        
    

