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

from layers import NestedUNet, NestedUNetIv3
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

    model = NestedUNet(2, 2)
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
    
    l = generator.length
    vl = generator.val_length

    criterion = BCEWithLogitsLoss(pos_weight = torch.FloatTensor([6.66666]).to(device))
    optimizer = optim.Adam(model.parameters(), lr = 0.001)
    scheduler = ReduceLROnPlateau(optimizer, factor = float(args.rl_factor), patience = int(args.n_plateau))

    min_val_loss = np.inf
    early_count = 0

    history = dict()
    history['epoch'] = []
    history['loss'] = []
    history['val_loss'] = []
    # ${define_extra_history}
    
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
            
            x = x.to(device)
            y = y.to(device)

            y_pred = model(x)

            loss = criterion(y_pred, y) # ${loss_change}
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

            # compute accuracy in CPU with sklearn
            y_pred = np.round(np.exp(y_pred.detach().cpu().numpy()))
            y = y.detach().cpu().numpy()

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

                x = x.to(device)
                y = y.to(device)
        
                y_pred = model(x)

                loss = criterion(y_pred, y)
                # compute accuracy in CPU with sklearn
                y_pred = np.round(np.exp(y_pred.detach().cpu().numpy()))
                y = y.detach().cpu().numpy()

                # append metrics for this epoch
                val_accs.append(accuracy_score(y.flatten(), y_pred.flatten()))
                val_losses.append(loss.detach().item())

        x = x.detach().cpu().numpy()
        for k in range(y_pred.shape[0]):
            fig, axes = plt.subplots(nrows = 3, ncols = 2, sharex = True)
            
            axes[0,0].imshow(x[k,0,:,:], cmap = 'gray')
            axes[0,1].imshow(x[k,1,:,:], cmap = 'gray')
            
            axes[1,0].imshow(y[k,0,:,:], cmap = 'gray')
            axes[1,1].imshow(y[k,1,:,:], cmap = 'gray')
            
            axes[2,0].imshow(y_pred[k,0,:,:], cmap = 'gray')
            axes[2,1].imshow(y_pred[k,1,:,:], cmap = 'gray')
            
            plt.savefig(os.path.join(os.path.join(args.odir, '{}_plots'.format(args.tag)), '{0:03d}.png'.format(k)), dpi = 100)
            plt.close()
            
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

