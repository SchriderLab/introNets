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

from layers import LexStyleNet
from data_loaders import H5DisDataGenerator
import h5py

import numpy as np
from sklearn.metrics import accuracy_score
import pandas as pd

import glob
from torchvision_mod_layers import resnet18, resnet34, resnet50

# use this format to tell the parsers
# where to insert certain parts of the script
# ${imports}

def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser()
    # my args
    parser.add_argument("--verbose", action = "store_true", help = "display messages")
    parser.add_argument("--idir", default = "None")

    parser.add_argument("--weights", default = "None", help = "weights to load (optional)")

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
from sklearn.metrics import accuracy_score, confusion_matrix
import pandas as pd
# use this format to tell the parsers
# where to insert certain parts of the script
# ${imports}

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import seaborn as sns

def cm_analysis(y_true, y_pred, filename, labels, ymap=None, figsize=(10,10)):
    """
    Generate matrix plot of confusion matrix with pretty annotations.
    The plot image is saved to disk.
    args: 
      y_true:    true label of the data, with shape (nsamples,)
      y_pred:    prediction of the data, with shape (nsamples,)
      filename:  filename of figure file to save
      labels:    string array, name the order of class labels in the confusion matrix.
                 use `clf.classes_` if using scikit-learn models.
                 with shape (nclass,).
      ymap:      dict: any -> string, length == nclass.
                 if not None, map the labels & ys to more understandable strings.
                 Caution: original y_true, y_pred and labels must align.
      figsize:   the size of the figure plotted.
    """
    if ymap is not None:
        y_pred = [ymap[yi] for yi in y_pred]
        y_true = [ymap[yi] for yi in y_true]
        labels = [ymap[yi] for yi in labels]
    cm = confusion_matrix(y_true, y_pred)
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
            elif c == 0:
                annot[i, j] = ''
            else:
                annot[i, j] = '%.1f%%\n%d' % (p, c)
    cm = pd.DataFrame(cm, index=labels, columns=labels)
    cm.index.name = 'Actual'
    cm.columns.name = 'Predicted'
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm, annot=annot, fmt='', ax=ax)
    plt.savefig(filename)
    plt.close()

from prettytable import PrettyTable

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params+=param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params

def main():
    args = parse_args()
    
    ifiles = dict()
    
    _ = glob.glob(os.path.join(args.idir, '*.hdf5'))
    
    for f in _:
        t = f.split('/')[-1].split('_')[0]
        
        ifiles[t] = f

    logging.info('reading data keys...')
    generator = H5DisDataGenerator(ifiles, batch_size = int(args.batch_size))

    logging.info('creating model...')
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    model = resnet34()
    model = model.to(device)
        
    print(model)
    d_model = count_parameters(model)
        
    if args.weights != "None":
        checkpoint = torch.load(args.weights, map_location = device)
        model.load_state_dict(checkpoint)
        
    model.train()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = 0.001)
    #scheduler = ReduceLROnPlateau(optimizer, factor = float(args.rl_factor), patience = int(args.n_plateau))

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
        for ij in range(generator.length):
            optimizer.zero_grad()
            x, y = generator.get_batch()
            
            if x is None:
                break

            x = x.to(device)
            
            y = y.to(device)
            y_pred = model(x)

            loss = criterion(y_pred, y) # ${loss_change}
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

            # compute accuracy in CPU with sklearn
            y_pred = y_pred.detach().cpu().numpy()
            y = y.detach().cpu().numpy()

            y_pred = np.argmax(y_pred, axis=1)

            # append metrics for this epoch
            accuracies.append(accuracy_score(y, y_pred))

            if (ij + 1) % 100 == 0:
                logging.info(
                    'root: Epoch {0}, step {3}: got loss of {1}, acc: {2}'.format(ix, np.mean(losses),
                                                                                  np.mean(accuracies), ij + 1))
            
            ij += 1

        model.eval()
        
        Y = []
        Y_pred = []

        val_losses = []
        val_accs = []
        for ij in range(generator.val_length):
            with torch.no_grad():
                x, y = generator.get_batch(True)
                
                if x is None:
                    break

                x = x.to(device)
                
                y = y.to(device)
                y_pred = model(x)

                loss = criterion(y_pred, y)
                # compute accuracy in CPU with sklearn
                y_pred = y_pred.detach().cpu().numpy()
                y = y.detach().cpu().numpy()

                y_pred = np.argmax(y_pred, axis=1)
                
                Y.extend(y)
                Y_pred.extend(y_pred)

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
            
            cm_analysis(Y, Y_pred, os.path.join(args.odir, '{}_best.png'.format(args.tag)), ['AB', 'BA', 'none'])

            early_count = 0
        else:
            early_count += 1

            # early stop criteria
            if early_count > int(args.n_early):
                break

        #scheduler.step(val_loss)
        generator.on_epoch_end()

        df = pd.DataFrame(history)
        df.to_csv(os.path.join(args.odir, '{}_history.csv'.format(args.tag)), index = False)
    
if __name__ == '__main__':
    main()