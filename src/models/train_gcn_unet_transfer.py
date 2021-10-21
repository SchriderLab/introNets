# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
import os
import argparse
import logging
import sys

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
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import pandas as pd
# use this format to tell the parsers
# where to insert certain parts of the script
# ${imports}

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from layers import GCNUNet_i1
from data_loaders import GCNDataGenerator

from torch_scatter import scatter_max
from scipy.special import expit

class TransferModel(nn.Module):
    def __init__(self, model, n = 150):
        super(TransferModel, self).__init__()
        
        self.res = list(model.children())[0]
        
        self.conv = list(model.children())[1]
        self.transform = nn.Sequential(*list(list(model.children())[-2].children())[:-1])
        
        self.out = nn.Linear(1024, 150)
        
        self.activation = nn.ReLU()
        
    def freeze(self):
        ### freeze all the layers of the network except for the last ones
        for param in self.res.parameters():
            param.requires_grad = False
            

    def unfreeze(self):
        ### unfreeze
        for param in self.res.parameters():
            param.requires_grad = True         
        
        
    def forward(self, x, edge_indices, batch):
        x = self.res(x, edge_indices)
        
        x_global = self.conv(torch.unsqueeze(x, 2))
        x_global = torch.squeeze(self.activation(x_global))
        
        x_global = scatter_max(x_global, batch, dim = 0)[0]
        
        x = torch.cat([x, x_global[batch]], dim = 1)
        
        x = self.transform(x)
        x = self.out(x)
        
        return x
    
    
    
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
    parser.add_argument("--n_early", default = "5")

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
        
    model = GCNUNet_i1(in_channels = 306, n_features = 306, n_classes = 1)        
    
    checkpoint = torch.load(args.weights, map_location = device)
    model.load_state_dict(checkpoint)
    
    model = TransferModel(model)
    model.to(device)    
    print(model)
        
    generator = GCNDataGenerator(args.idir, batch_size = int(args.batch_size), seg = True)
    criterion = nn.BCEWithLogitsLoss(pos_weight = torch.FloatTensor([0.6713357505900737]).to(device))
    
    optimizer = optim.Adam(model.parameters(), lr = 0.00001)
    early_count = 0
    #scheduler = ReduceLROnPlateau(optimizer, factor = float(args.rl_factor), patience = int(args.n_plateau))

    history = dict()
    history['loss'] = []
    history['epoch'] = []
    history['val_loss'] = []

    model.freeze()

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
            except Exception as e:
                print(e)
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
            y_pred = np.round(expit(y_pred.detach().cpu().numpy().flatten()))
            y = np.round(y.detach().cpu().numpy().flatten())
    
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
                    
                    if x.shape[0] != y.shape[0]:
                        continue
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
                y_pred = expit(y_pred.detach().cpu().numpy().flatten())
                y = y.detach().cpu().numpy().flatten()
                
                val_accs.append(accuracy_score(np.round(y), np.round(y_pred)))
                
                Y.extend(y)
                Y_pred.extend(np.round(y_pred))
        
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
            
            cm_analysis(Y, Y_pred, os.path.join(args.odir, 'best.png'), ['native', 'introgressed'])
    
            early_count = 0
        else:
            early_count += 1
    
            # early stop criteria
            if early_count > int(args.n_early):
                break
    
        generator.on_epoch_end()
        
        df = pd.DataFrame(history)
        df.to_csv(os.path.join(args.odir, '{}_history.csv'.format(args.tag)), index = False)
        
    model.unfreeze()
    print('training unfrozen...')
    for ix in range(int(args.n_epochs)):
        model.train()
    
        losses = []
        accuracies = []
    
        for ij in range(generator.length):
            optimizer.zero_grad()
            
            try:
                x, y, edges, batch = generator.get_batch()
            except Exception as e:
                print(e)
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
            y_pred = np.round(expit(y_pred.detach().cpu().numpy().flatten()))
            y = np.round(y.detach().cpu().numpy().flatten())
    
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
                    
                    if x.shape[0] != y.shape[0]:
                        continue
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
                y_pred = expit(y_pred.detach().cpu().numpy().flatten())
                y = y.detach().cpu().numpy().flatten()
                
                val_accs.append(accuracy_score(np.round(y), np.round(y_pred)))
                
                Y.extend(y)
                Y_pred.extend(np.round(y_pred))
        
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
            
            cm_analysis(Y, Y_pred, os.path.join(args.odir, 'best.png'), ['native', 'introgressed'])
    
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
        
    

