# -*- coding: utf-8 -*-
import os
import argparse
import logging

import glob
from runAndParseSlim import readSampleOutFromSlimRun, buildMutationPosMapping, removeMonomorphic, buildPositionsList, writeIntrogressedAlleles
import gzip

def emitMsEntry(positions, segsites, haps, numReps, out, isFirst=True):
    if isFirst:
        out.write("slim {} {}\n".format(len(haps), numReps))
        out.write("blah\n")
    out.write("\n//\n")
    out.write("segsites: {}\n".format(segsites))
    out.write("positions: "+ " ".join([str(x) for x in positions]) + "\n")
    for hap in haps:
        out.write("".join([str(x) for x in hap]) + "\n")

def emitMsEntry(positions, segsites, haps, numReps, isFirst=True):
    if isFirst:
        print("slim {} {}".format(len(haps), numReps))
        print("blah")
    print("\n//")
    print("segsites: {}".format(segsites))
    print("positions: "+ " ".join([str(x) for x in positions]))
    for hap in haps:
        print("".join([str(x) for x in hap]))

# use this format to tell the parsers
# where to insert certain parts of the script
# ${imports}

def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser()
    # my args
    parser.add_argument("--verbose", action = "store_true", help = "display messages")
    parser.add_argument("--idir", default = "/proj/dschridelab/miscShared/toDylan/bgsTestSlimOutput")
    parser.add_argument("--n_per_pop", default = "64")
    parser.add_argument("--phys_len", default = "1000000")


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
    
    ifiles = glob.glob(os.path.join(args.idir, '*.out.gz'))
    physLen = int(args.phys_len)
    
    for ifile in ifiles:
        introgLogFileName = os.path.join(args.odir, ifile.split('/')[-1].replace('.out.gz', '.log'))
        msOutFileName = os.path.join(args.odir, ifile.split('/')[-1].replace('.out.gz', '.log'))

        ifile = gzip.open(ifile, 'r')
        output = ifile.readlines()
        
        outF = open(introgLogFileName, "wt")
        outF_ms = open(msOutFileName, "wt")

        mutations, genomes, introgressedAlleles = readSampleOutFromSlimRun(output, 1, int(args.n_per_pop))
        newMutLocs = []
        for mutPos in mutations:
            if len(mutations[mutPos]) == 1:
                mutId = list(mutations[mutPos].keys())[0]
                newMutLocs.append((mutPos, mutId))
            else:
                for mutId in mutations[mutPos]:
                    newMutLocs.append((mutPos, mutId))
    
        allMuts = buildMutationPosMapping(newMutLocs, physLen)
        polyMuts = removeMonomorphic(allMuts, genomes)
        positions = buildPositionsList(polyMuts)
        haps = []
        for i in range(len(genomes)):
            haps.append([0]*len(polyMuts))
    
        for i in range(len(genomes)):
            for locI, loc, locCont, mutId in polyMuts:
                if mutId in genomes[i]:
                    haps[i][locI] = 1
  
        emitMsEntry(positions, len(polyMuts), haps, 1, outF_ms, isFirst=True)
        writeIntrogressedAlleles(0, introgressedAlleles, physLen, outF)
    
        outF.close()
        outF_ms.close()
        
        os.system('gzip {} {}'.format(introgLogFileName, msOutFileName))

    # ${code_blocks}

if __name__ == '__main__':
    main()


