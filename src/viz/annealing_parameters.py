# -*- coding: utf-8 -*-
import os
import argparse
import logging

import matplotlib
matplotlib.use("Agg")

bounds = dict()
bounds[0] = (1., 150.) # theta
bounds[1] = (0.01, 0.35) # theta_rho
bounds[2] = (0.1, 200.) # nu_ab
bounds[3] = (0.001, 1.0) # nu_ba
bounds[4] = (0.01, None) # migTime
bounds[5] = (0.01, 1.0) # migProb
bounds[6] = (0.01, 15.) # T
bounds[7] = (-10., 10.) # alpha1
bounds[8] = (-20, 10.) # alpha2

def parameters_df(df, ix, thetaOverRho = 0.2):
    u = 3.5e-9
    L = 10000
    
    ll, aic, Nref, nu1_0, nu2_0, nu1, nu2, T, Nref_m12, Nref_m21 = df[ix]
    
    nu1_0 /= Nref
    nu2_0 /= Nref
    nu1 /= Nref
    nu2 /= Nref
    
    T /= (4*Nref / 15.)
    
    alpha1 = np.log(nu1/nu1_0)/T
    alpha2 = np.log(nu2/nu2_0)/T
    
    theta = 4 * Nref * u * L
    rho = theta / thetaOverRho
    
    return ll, theta, rho, nu1, nu2, alpha1, alpha2, T

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
    
    # T
    p[6] = (p[6] - bounds[6][0]) / (bounds[6][1] - bounds[6][0])

    return p

import matplotlib.pyplot as plt
import numpy as np

# use this format to tell the parsers
# where to insert certain parts of the script
# ${imports}

def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser()
    # my args
    parser.add_argument("--verbose", action = "store_true", help = "display messages")
    parser.add_argument("--ifile", default = "None")
    parser.add_argument("--idir", default = "None")
    
    parser.add_argument("--normalize", action = "store_true")    
    # ${args}

    parser.add_argument("--odir", default = "None")
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
    
    plot_titles = ['theta', 'rho', 'nu1', 'nu2', 'alpha1', 'alpha2', 'T']
    
    if args.ifile != "None":
        df = np.loadtxt(args.ifile)
        
        ll = []
        theta = []
        rho = []
        nu1 = []
        nu2 = []
        alpha1 = []
        alpha2 = []
        T = []
        
        for ix in range(df.shape[0]):
            ll_, theta_, rho_, nu1_, nu2_, alpha1_, alpha2_, T_ = parameters_df(df, ix)
            
            if ll_ > -2000:
                ll.append(ll_)
                theta.append(theta_)
                rho.append(rho_)
                nu1.append(nu1_)
                nu2.append(nu2_)
                alpha1.append(alpha1_)
                alpha2.append(alpha2_)
                T.append(T_)
                
        print('theta: ==================')
        print('mean: {}'.format(np.mean(theta)))
        print('min: {}'.format(np.min(theta)))
        print('max: {}'.format(np.max(theta)))
        print('var / mean: {}'.format(np.var(theta) / np.mean(theta)))
        
        print('nu1: ==================')
        print('mean: {}'.format(np.mean(nu1)))
        print('min: {}'.format(np.min(nu1)))
        print('max: {}'.format(np.max(nu1)))
        print('var / mean: {}'.format(np.var(nu1) / np.mean(nu1)))
        
        print('nu2: ==================')
        print('mean: {}'.format(np.mean(nu2)))
        print('min: {}'.format(np.min(nu2)))
        print('max: {}'.format(np.max(nu2)))
        print('var / mean: {}'.format(np.var(nu2) / np.mean(nu2)))
        
        print('alpha1: ==================')
        print('mean: {}'.format(np.mean(alpha1)))
        print('min: {}'.format(np.min(alpha1)))
        print('max: {}'.format(np.max(alpha1)))
        print('var / mean: {}'.format(np.var(alpha1) / np.mean(alpha1)))
        
        print('alpha2: ==================')
        print('mean: {}'.format(np.mean(alpha2)))
        print('min: {}'.format(np.min(alpha2)))
        print('max: {}'.format(np.max(alpha2)))
        print('var / mean: {}'.format(np.var(alpha2) / np.mean(alpha2)))
        
        print('T: ==================')
        print('mean: {}'.format(np.mean(T)))
        print('min: {}'.format(np.min(T)))
        print('max: {}'.format(np.max(T)))
        print('var / mean: {}'.format(np.var(T) / np.mean(T)))
        
        fig = plt.figure(figsize=(10, 2), dpi=80)
        ax = fig.add_subplot(171)
        ax.set_title('theta')
        ax.hist(theta, bins = 25)
        
        ax = fig.add_subplot(172)
        ax.set_title('rho')
        ax.hist(rho, bins = 25)
        
        ax = fig.add_subplot(173)
        ax.set_title('nu1')
        ax.hist(nu1, bins = 25)
        
        ax = fig.add_subplot(174)
        ax.set_title('nu2')
        ax.hist(nu2, bins = 25)
        
        ax = fig.add_subplot(175)
        ax.set_title('alpha1')
        ax.hist(alpha1, bins = 25)
        
        ax = fig.add_subplot(176)
        ax.set_title('alpha2')
        ax.hist(alpha2, bins = 25)
        
        ax = fig.add_subplot(177)
        ax.set_title('T')
        ax.hist(T, bins = 25)
        
        plt.savefig(os.path.join(args.odir, 'raw_hists.png'))
        plt.close()
        
        P = np.array([theta, rho, nu1, nu2, alpha1, alpha2, T]).T
        
        for k in range(len(P)):
            P[k] = normalize(P[k])
            
        P = P.T
        theta = P[0,:]
        rho = P[1,:]
        nu1 = P[2,:]
        nu2 = P[3,:]
        alpha1 = P[4,:]
        alpha2 = P[5,:]
        T = P[6,:]
        
        fig = plt.figure(figsize=(10, 2), dpi=80)
        ax = fig.add_subplot(171)
        ax.set_title('theta')
        ax.hist(theta, bins = 25)
        
        ax = fig.add_subplot(172)
        ax.set_title('rho')
        ax.hist(rho, bins = 25)
        
        ax = fig.add_subplot(173)
        ax.set_title('nu1')
        ax.hist(nu1, bins = 25)
        
        ax = fig.add_subplot(174)
        ax.set_title('nu2')
        ax.hist(nu2, bins = 25)
        
        ax = fig.add_subplot(175)
        ax.set_title('alpha1')
        ax.hist(alpha1, bins = 25)
        
        ax = fig.add_subplot(176)
        ax.set_title('alpha2')
        ax.hist(alpha2, bins = 25)
        
        ax = fig.add_subplot(177)
        ax.set_title('T')
        ax.hist(T, bins = 25)
        
        plt.savefig(os.path.join(args.odir, 'norm_hists.png'))
        plt.close()
        
        plt.hist(ll, bins = 20)
        plt.savefig(os.path.join(args.odir, 'll.png'), dpi = 100)
        plt.close()
        
        
        

    # ${code_blocks}

if __name__ == '__main__':
    main()

