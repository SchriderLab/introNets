# -*- coding: utf-8 -*-
import os
import argparse
import logging

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

    idirs = [os.path.join(args.idir, u) for u in os.listdir(args.idir) if not 'training' in u]
    
    train, val = pickle.load(open(args.ifile, 'rb'))
    
    for idir in idirs:
        print('working on {}'.format(idir))
        odir = os.path.join(args.odir, idir.split('/')[-1] + '_val')
        os.system('mkdir -p {}')
        
        for ifile in val:
            os.system('cp {0} {1}'.format(os.path.join(idir, ifile), odir))
            
        

if __name__ == '__main__':
    main()

