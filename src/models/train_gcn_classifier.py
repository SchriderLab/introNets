# -*- coding: utf-8 -*-
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

from layers import NestedUNet
from data_loaders import H5UDataGenerator
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
from layers import GCNClassifier
from data_loaders import GCNDisDataGenerator

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

    parser.add_argument("--batch_size", default = "16")
    
    parser.add_argument("--seg", action = "store_true")
    parser.add_argument("--loss", default = "bce")
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


    model = GCNClassifier()
    if len(device_strings) > 1:
        model = nn.DataParallel(model, device_ids = list(map(int, args.devices.split(',')))).to(device)
        model = model.to(device)
    else:
        model = model.to(device)
        
    if args.weights != "None":
        checkpoint = torch.load(args.weights, map_location = device)
        model.load_state_dict(checkpoint)
        
    generator = GCNDisDataGenerator(args.idir, batch_size = int(args.batch_size), seg = args.seg)
    
    criterion = nn.CrossEntropyLoss()
    
    optimizer = optim.Adam(model.parameters(), lr = 0.001)
    early_count = 0
    #scheduler = ReduceLROnPlateau(optimizer, factor = float(args.rl_factor), patience = int(args.n_plateau))

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
            
            try:
                x, y, edges, batch = generator.get_batch()
            except:
                break
            
            x = x.to(device)
            y = y.to(device)
            edges = [u.to(device) for u in edges]
            batch = batch.to(device)

            y_pred = model(x, edges, batch)
            #print(y.shape, y_pred.shape)


            loss = criterion(y_pred, y) # ${loss_change}
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

            # compute accuracy in CPU with sklearn
            y_pred = np.round(y_pred.detach().cpu().numpy().flatten())
            y = np.argmax(y.detach().cpu().numpy(), axis = 1).flatten()

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
                try:
                    x, y, edges, batch = generator.get_batch(val = True)
                except:
                    break

                x = x.to(device)
                y = y.to(device)
                edges = [u.to(device) for u in edges]
                batch = batch.to(device)
    
                y_pred = model(x, edges, batch)

                loss = criterion(y_pred, y)
                val_losses.append(loss.detach().item())
                
                # compute accuracy in CPU with sklearn
                y_pred = y_pred.detach().cpu().numpy().flatten()
                y = np.argmax(y.detach().cpu().numpy(), axis = 1).flatten()
                
                val_accs.append(accuracy_score(y, y_pred))
                
                Y.extend(y)
                Y_pred.extend(y_pred)
        
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
            
            Y = np.array(Y)
            Y_pred = np.array(Y_pred)
            
            ix_ = list(np.random.choice(range(len(Y)), 1000, replace = False))
            
            fig, axes = plt.subplots(ncols = 2)
            axes[0].scatter(Y[ix_], Y_pred[ix_], alpha = 0.1)
            axes[0].plot([0, 1], [0, 1], color = 'k')
            
            axes[1].hist(Y_pred[ix_] - Y[ix_], bins = 35)
            
            plt.savefig(os.path.join(args.odir, '{}_best.png'.format(args.tag)))
            plt.close()

            early_count = 0
        else:
            early_count += 1

            # early stop criteria
            if early_count > int(args.n_early):
                break

        generator.on_epoch_end()
        
        df = pd.DataFrame(history)
        df.to_csv(os.path.join(args.odir, '{}_history.csv'.format(args.tag)), index = False)

if __name__ == '__main__':
    main()
        
    

