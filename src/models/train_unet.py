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

from layers import NestedUNet, NestedUNetV2
from data_loaders import H5UDataGenerator
import h5py

import numpy as np
from sklearn.metrics import accuracy_score
import pandas as pd
import pickle
# use this format to tell the parsers
# where to insert certain parts of the script
# ${imports}

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
# pos weights for the different iterations:
## 000: 5.146955817634787
## 001: 5.469970200977819
## 002: 6.25023282070848

# i2
## 000: 8.379942663085659
## 001: 4.116404380681516
## 002: 4.449054245190725
## 003: 5.2208065886452


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
    parser.add_argument("--loss", default = "bce")
    
    # for AO to BF we had: 
    parser.add_argument("--pos_weight", default = "8.379942663085659")
    parser.add_argument("--n_classes", default = "1")

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

    model = NestedUNetV2(int(args.n_classes), 2)
    if len(device_strings) > 1:
        model = nn.DataParallel(model, device_ids = list(map(int, args.devices.split(',')))).to(device)
        model = model.to(device)
    else:
        model = model.to(device)
        
    if args.weights != "None":
        checkpoint = torch.load(args.weights, map_location = device)
        model.load_state_dict(checkpoint)

    # define the generator
    print('reading data keys...')
    generator = H5UDataGenerator(h5py.File(args.ifile, 'r'), batch_size = int(args.batch_size))
    val_keys = generator.val_keys
    
    # save them for later
    pickle.dump(val_keys, open(os.path.join(args.odir, '{}_val_keys.pkl'.format(args.tag)), 'wb'))
    
    l = generator.length
    vl = generator.val_length

    criterion = BCEWithLogitsLoss(pos_weight = torch.FloatTensor([float(args.pos_weight)]).to(device))
    optimizer = optim.Adam(model.parameters(), lr = 0.001)
    
    decayRate = 0.96
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer = optimizer, gamma=decayRate)

    min_val_loss = np.inf
    early_count = 0

    history = dict()
    history['epoch'] = []
    history['loss'] = []
    history['val_loss'] = []
    
    # for plotting
    os.system('mkdir -p {}'.format(os.path.join(args.odir, '{}_plots'.format(args.tag))))

    print('training...')
    for ix in range(int(args.n_epochs)):
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

        lr_scheduler.step()
        generator.on_epoch_end()

        df = pd.DataFrame(history)
        df.to_csv(os.path.join(args.odir, '{}_history.csv'.format(args.tag)), index = False)


if __name__ == '__main__':
    main()
# -*- coding: utf-8 -*-

