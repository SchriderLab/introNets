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
from data_loaders import H5DisDataGenerator, DisDataGenerator
import h5py

import numpy as np
from sklearn.metrics import accuracy_score
import pandas as pd
# use this format to tell the parsers
# where to insert certain parts of the script
# ${imports}



def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser()
    # my args
    parser.add_argument("--verbose", action = "store_true", help = "display messages")
    parser.add_argument("--idir_sims", default = "None")
    parser.add_argument("--idir_real", default = "None")

    parser.add_argument("--weights", default = "None", help = "weights to load (optional)")

    parser.add_argument("--devices", default = "0")
    parser.add_argument("--n_plateau", default = "5")
    parser.add_argument("--rl_factor", default = "0.5")
    parser.add_argument("--n_epochs", default = "10")
    parser.add_argument("--n_early", default = "10")

    parser.add_argument("--batch_size", default = "16")
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

    model = PermInvariantClassifier()
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
    generator = DisDataGenerator(args.idir_sims, args.idir_real, batch_size = int(args.batch_size))

    criterion = NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr = 0.001)
    scheduler = ReduceLROnPlateau(optimizer, factor = float(args.rl_factor), patience = int(args.n_plateau))

    min_val_loss = np.inf
    early_count = 0

    history = dict()
    history['epoch'] = []
    history['loss'] = []
    history['val_loss'] = []
    # ${define_extra_history}

    print('training...')
    for ix in range(int(args.n_epochs)):
        model.train()

        losses = []
        accuracies = []

        ij = 0
        while not generator.done:
            optimizer.zero_grad()
            x1, x2, y = generator.get_batch()
            
            if x1 is None:
                break

            x1 = x1.to(device)
            x2 = x2.to(device)
            
            y = y.to(device)

            y_pred = model(x1, x2)

            loss = criterion(y_pred, y) # ${loss_change}
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

            # compute accuracy in CPU with sklearn
            y_pred = np.exp(y_pred.detach().cpu().numpy())
            y = y.detach().cpu().numpy()

            y_pred = np.argmax(y_pred, axis=1)

            # append metrics for this epoch
            accuracies.append(accuracy_score(y, y_pred))

            if (ij + 1) % 100 == 0:
                logging.info(
                    'root: Epoch {0}, step {3}: got loss of {1}, acc: {2}'.format(ix, np.mean(losses),
                                                                                  np.mean(accuracies), ij + 1))
                break
            
            ij += 1

        model.eval()

        val_losses = []
        val_accs = []
        while not generator.val_done:
            with torch.no_grad():
                x1, x2, y = generator.get_val_batch()
                
                if x1 is None:
                    break

                x1 = x1.to(device)
                x2 = x2.to(device)
                
                y = y.to(device)
    
                y_pred = model(x1, x2)

                loss = criterion(y_pred, y)
                # compute accuracy in CPU with sklearn
                y_pred = np.exp(y_pred.detach().cpu().numpy())
                y = y.detach().cpu().numpy()

                y_pred = np.argmax(y_pred, axis=1)

                # append metrics for this epoch
                val_accs.append(accuracy_score(y, y_pred))
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

        scheduler.step(val_loss)
        generator.on_epoch_end()

        df = pd.DataFrame(history)
        df.to_csv(os.path.join(args.odir, '{}_history.csv'.format(args.tag)), index = False)


if __name__ == '__main__':
    main()
# -*- coding: utf-8 -*-

