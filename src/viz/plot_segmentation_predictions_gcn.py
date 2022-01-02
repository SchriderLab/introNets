# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import logging
import argparse
import os

import torch

import h5py

import sys
sys.path.insert(0, os.path.join(os.getcwd(), 'src/models'))

from layers_1d import GATRelateCNetV2, GCNUNet_delta
from data_loaders import H5UDataGenerator, GCNDataGeneratorH5

from scipy.special import expit
from sklearn.decomposition import PCA

def transform_im(pca, x):
    # reshape to (N, channels) for transform
    x = x.transpose(0, 2, 3, 1)
    shape = list(x.shape)
    
    x = x.reshape(np.product(list(shape[:-1])), shape[-1])
    
    # do the linear transform
    x = pca.transform(x)
    
    # reshape back to batch,h,w,channel
    x = x.reshape(*shape)
    
    return x
    

def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser()
    # my args
    parser.add_argument("--verbose", action = "store_true", help = "display messages")
    parser.add_argument("--weights", default = "None") # weights of the pre-trained model
    parser.add_argument("--ifile", default = "None")
    
    parser.add_argument("--odir", default = "None")
    parser.add_argument("--n_samples", default = "4")
    
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
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    model = GCNUNet_delta()
    if args.weights != "None":
        checkpoint = torch.load(args.weights, map_location = device)
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()
    
    generator = GCNDataGeneratorH5(args.ifile,
                                 batch_size = 4)
    counter = 0
    
    xs_down = []
    xs_up = []
    
    logging.info('plotting initial predictions and gathering intermediates...')
    
    for ix in range(int(args.n_samples)):
        with torch.no_grad():
            x, y, edges, edge_attr, batch = generator.get_batch(True)
            
            print(x.shape, y.shape)

            x = x.to(device)
            y = y.to(device)
            
            edges = edges.to(device)
            batch = batch.to(device)
            edge_attr = edge_attr.to(device)
            
            y_pred, xs_down_, xs_up_ = model(x, edges, edge_attr, batch, return_intermediates = True)
            
            print(y_pred.shape)
            
            x = x.detach().cpu().numpy()
            y = y.detach().cpu().numpy()
            y_pred = y_pred.detach().cpu().numpy()
            batch = batch.detach().cpu().numpy()
            
            xs_down_ = [u.detach().cpu().numpy() for u in xs_down_]
            xs_up_ = [u.detach().cpu().numpy() for u in xs_up_]
            
            xs_down.append(xs_down_)
            xs_up.append(xs_up_)
            
            for k in list(set(batch)):                
                fig = plt.figure(figsize=(16, 6))
                ax0 = fig.add_subplot(141)
                
                ax0.imshow(x[k,0,:,:], cmap = 'gray')
                ax0.set_title('pop A')
                
                ax2 = fig.add_subplot(142)
                ax2.imshow(y[k,:], cmap = 'gray')
                ax2.set_title('pop B (y)')
                
                ax3 = fig.add_subplot(143)
                ax3.imshow(np.round(expit(y_pred[k,:])), cmap = 'gray')
                ax3.set_title('pop B (pred)')
                
                ax4 = fig.add_subplot(144)
                im = ax4.imshow(expit(y_pred[k,:]))
                fig.colorbar(im, ax = ax4)
                
                plt.savefig(os.path.join(args.odir, '{0:04d}_pred.png'.format(counter)), dpi = 100)
                counter += 1
                plt.close()
                
                
    logging.info('fitting down PCAs...')
    # compute the PCA of each pixel space -> 3
    # down side of the U
    channels_down = dict()
    for k in range(len(xs_down[0])):
        channels_down[k] = []
    
    for j in range(len(xs_down)):
        x = xs_down[j]
        
        for k in range(len(x)):
            channels_down[k].append(x[k].transpose(0,2,3,1))
            shape = list(channels_down[k][-1].shape)
            
            channels_down[k][-1] = channels_down[k][-1].reshape(np.product(shape[:-1]), shape[-1])
        
    pcas_down = []
    
    for key in sorted(channels_down.keys()):
        channels_down[key] = np.array(channels_down[key], dtype = np.float32)
        print(channels_down[key].shape)
        
        
        pca = PCA(3)
        pca.fit(channels_down[key])
        
        pcas_down.append(pca)
        
    del xs_down
        
    logging.info('fitting up pcas...')
    # for the other side of the U (up)
    channels_up = dict()
    for k in range(len(xs_up[0])):
        channels_up[k] = []
    
    for j in range(len(xs_up)):
        x = xs_up[j]
        
        for k in range(len(x)):
            channels_up[k].append(x[k].transpose(0,2,3,1))
            shape = list(channels_up[k][-1].shape)
            
            channels_up[k][-1] = channels_up[k][-1].reshape(np.product(shape[:-1]), shape[-1])
        
    pcas_up = []
    
    for key in sorted(channels_up.keys()):
        channels_up[key] = np.array(channels_up[key], dtype = np.float32)
        
        pca = PCA(3)
        pca.fit(channels_down[key])
        
        pcas_up.append(pca)
        
    del xs_up
    
    logging.info('plotting with intermediates...')
    for ix in range(int(args.n_samples)):
        with torch.no_grad():
            x, y, edges, edge_attr, batch = generator.get_batch(True)
            
            print(x.shape, y.shape)

            x = x.to(device)
            y = y.to(device)
            
            edges = edges.to(device)
            batch = batch.to(device)
            edge_attr = edge_attr.to(device)
            
            y_pred, xs_down_, xs_up_ = model(x, edges, edge_attr, batch, return_intermediates = True)
            
            print(y_pred.shape)
            
            x = x.detach().cpu().numpy()
            y = y.detach().cpu().numpy()
            y_pred = y_pred.detach().cpu().numpy()
            batch = batch.detach().cpu().numpy()
            
            xs_down_ = [u.detach().cpu().numpy() for u in xs_down_]
            xs_up_ = [u.detach().cpu().numpy() for u in xs_up_]
            
            # for each one transform the n-channel space to 3-channel:
            xs_down_ = [transform_im(pcas_down[k], xs_down_[k]) for k in range(len(xs_down_))]
            xs_up_ = [transform_im(pcas_down[k], xs_down_[k]) for k in range(len(xs_up_))]
            
            for k in list(set(batch)):                
                fig = plt.figure(figsize=(16, 12))
          
                ax1 = fig.add_subplot(451)
                ax1.imshow(x[k,0,:,:])
                ax1.set_title('X')
                
                ##### down side of the U
                # -----------------------
                ax0 = fig.add_subplot(4, 4, 5)
                ax0.imshow(xs_down_[0][k,:,:,:])
                ax0.set_title('down transforms')
                
                ax0 = fig.add_subplot(446)                
                ax0.imshow(xs_down_[1][k,:,:,:])
                
                ax0 = fig.add_subplot(447)                
                ax0.imshow(xs_down_[2][k,:,:,:])
                
                ax0 = fig.add_subplot(448)                
                ax0.imshow(xs_down_[3][k,:,:,:])
                
                ##### up side of the U
                # ---------------------
                ax0 = fig.add_subplot(449)
                ax0.imshow(xs_up_[-1][k,:,:,:])
                ax0.set_title('down transforms')
                
                ax0 = fig.add_subplot(4, 4, 10)                
                ax0.imshow(xs_up_[-2][k,:,:,:])
                
                ax0 = fig.add_subplot(4, 4, 11)                
                ax0.imshow(xs_up_[-3][k,:,:,:])
                
                ax0 = fig.add_subplot(4, 4, 12)                
                ax0.imshow(xs_up_[-4][k,:,:,:])
                
                ax2 = fig.add_subplot(4, 4, 13)
                ax2.imshow(y[k,:], cmap = 'gray')
                ax2.set_title('pop B (y)')
                
                ax3 = fig.add_subplot(4, 4, 14)
                ax3.imshow(np.round(expit(y_pred[k,:])), cmap = 'gray')
                ax3.set_title('pop B (pred)')
                
                ax4 = fig.add_subplot(4, 4, 15)
                im = ax4.imshow(expit(y_pred[k,:]))
                fig.colorbar(im, ax = ax4)
                
                plt.savefig(os.path.join(args.odir, '{0:04d}_pred_inters.png'.format(counter)), dpi = 100)
                counter += 1
                plt.close()
        
    
    

    
if __name__ == '__main__':
    main()
