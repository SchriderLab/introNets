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
from scipy.special import expit

from layers import NestedUNet
from data_loaders import H5UDataGenerator
import configparser
import h5py

import numpy as np
from sklearn.metrics import accuracy_score
import pandas as pd
import pickle

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from prettytable import PrettyTable

import time

def count_parameters(model, f):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params+=param
    print(table, file = f)
    print(f"Total Trainable Params: {total_params}", file = f)
    return total_params

def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser()
    # my args
    parser.add_argument("--verbose", action = "store_true", help = "display messages")
    parser.add_argument("--ifile", default = "BA_10e6_seriated.hdf5")
    parser.add_argument("--config", default = "None", help = "pythonic config file for model and training parameters")
    parser.add_argument("--val_keys", default = "None", help = "a pkl file of the validation keys in the hdf5 input file.  if not provided one will be generated and saved with the training output")

    parser.add_argument("--weights", default = "None", help = "weights to load (optional)")
    parser.add_argument("--device", default = "0", help = "index of the GPU to use if available")
    parser.add_argument("--n_early", default = "10")

    parser.add_argument("--odir", default = "training_output", help = "folder to store the training results, weights, etc.")
    parser.add_argument("--tag", default = "test", help = "tag naming the output files")
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

    return args

def main():
    start_time = time.time()
    
    args = parse_args()

    if torch.cuda.is_available():
        device = torch.device('cuda:{}'.format(args.device))
    else:
        device = torch.device('cpu')
        
    config = configparser.ConfigParser()
    config.read(args.config)
    
    log_file = open(os.path.join(args.odir, '{}.log'.format(args.tag)), 'w')
    config.write(log_file)
    log_file.write('\n')
    
    model = NestedUNet(int(config.get('model_params', 'n_classes')), 2)
    print(model, file = log_file)
    model = model.to(device)
    
    try:
        n_steps = int(config.get('training_params', 'n_steps'))
    except:
        n_steps = None
    
    count_parameters(model, log_file)
        
    if args.weights != "None":
        checkpoint = torch.load(args.weights, map_location = device)
        model.load_state_dict(checkpoint)

    # define the generator
    print('reading data keys...')
    generator = H5UDataGenerator(h5py.File(args.ifile, 'r'), batch_size = int(config.get('training_params', 'batch_size')), 
                                 label_noise = float(config.get('training_params', 'label_noise')), label_smooth = config.getboolean('training_params', 'label_smooth'))
    val_keys = generator.val_keys
    
    # save them for later
    pickle.dump(val_keys, open(os.path.join(args.odir, '{}_val_keys.pkl'.format(args.tag)), 'wb'))
    
    if n_steps is None:
        l = generator.length
    else:
        l = n_steps
    vl = generator.val_length

    criterion = BCEWithLogitsLoss(pos_weight = torch.FloatTensor([float(config.get('training_params', 'pos_bce_logits_weight'))]).to(device))
    optimizer = optim.Adam(model.parameters(), lr = float(config.get('training_params', 'lr')))
    
    if config.get('training_params', 'schedule') == 'exponential':
        decayRate = float(config.get('training_params', 'exp_decay'))
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer = optimizer, gamma=decayRate)
    else:
        lr_scheduler = None

    min_val_loss = np.inf
    early_count = 0

    history = dict()
    history['epoch'] = []
    history['loss'] = []
    history['val_loss'] = []
    history['val_acc'] = []
    history['epoch_time'] = []

    print('training...')
    for ix in range(int(config.get('training_params', 'n_epochs'))):
        t0 = time.time()
        
        model.train()

        losses = []
        accuracies = []

        for ij in range(l):
            optimizer.zero_grad()
            x, y = generator.get_batch()
            
            y = torch.squeeze(y)
            
            x = x.to(device)
            y = y.to(device)

            y_pred = model(x)

            loss = criterion(y_pred, y) # ${loss_change}
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

            # compute accuracy in CPU with sklearn
            y_pred = np.round(expit(y_pred.detach().cpu().numpy().flatten()))
            y = np.round(y.detach().cpu().numpy().flatten())

            # append metrics for this epoch
            accuracies.append(accuracy_score(y.flatten(), y_pred.flatten()))

            if (ij + 1) % 100 == 0:
                logging.info(
                    'root: Epoch {0}, step {3}: got loss of {1}, acc: {2}'.format(ix, np.mean(losses),
                                                                                  np.mean(accuracies), ij + 1))

        model.eval()

        val_losses = []
        val_accs = []
        for step in range(vl):
            with torch.no_grad():
                x, y = generator.get_val_batch()
                
                y = torch.squeeze(y)

                x = x.to(device)
                y = y.to(device)
        
                y_pred = model(x)

                loss = criterion(y_pred, y)
                # compute accuracy in CPU with sklearn
                y_pred = np.round(expit(y_pred.detach().cpu().numpy().flatten()))
                y = np.round(y.detach().cpu().numpy().flatten())

                # append metrics for this epoch
                val_accs.append(accuracy_score(y.flatten(), y_pred.flatten()))
                val_losses.append(loss.detach().item())
            
        val_loss = np.mean(val_losses)

        logging.info(
            'root: Epoch {0}, got val loss of {1}, acc: {2} '.format(ix, val_loss, np.mean(val_accs)))

        history['epoch'].append(ix)
        history['loss'].append(np.mean(losses))
        history['val_loss'].append(np.mean(val_losses))
        
        e_time = time.time() - t0
        
        history['epoch_time'].append(e_time)
        history['val_acc'].append(np.mean(val_accs))

        val_loss = np.mean(val_losses)
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            logging.info('saving weights...')
            torch.save(model.state_dict(), os.path.join(args.odir, '{0}.weights'.format(args.tag)))

            early_count = 0
        else:
            early_count += 1

            # early stop criteria
            if early_count > int(args.n_early):
                break

        if lr_scheduler is not None:
            lr_scheduler.step()
        
        generator.on_epoch_end()

        df = pd.DataFrame(history)
        df.to_csv(os.path.join(args.odir, '{}_history.csv'.format(args.tag)), index = False)

    # benchmark the time to train
    total = time.time() - start_time
    log_file.write('training complete! \n')
    log_file.write('training took {} seconds... \n'.format(total))
    log_file.close()

if __name__ == '__main__':
    main()
# -*- coding: utf-8 -*-


