# -*- coding: utf-8 -*-
import sys, os, random, subprocess
import numpy as np

"""
example running 10 reps of 10kb sims with migration from pop 1 to pop 2 (change donorPop to 2 to reverse the order):
python runAndParseSlim.py introg.slim 10 10000 1 introg.log 1> test.ms 2> test.log
output:
introg.log contains for each individual a list of chunks of the chromosome that 
test.ms contains the output in ms-style format (but with segsite positions set to be discrete rather than continuous);
note that it is easy to change this to save to an npz file if you prefer: the 'positions' and 'haps' args to the
emitMsEntry function just need to be aggregated, converted to numpy arrays, and then saved to an npz at the end of
the script.
test.lot contains some other information about each rep (including the values of some of the randomly drawn parameters)
"""

def parseFixations(fixationLines):
    fixations = []
    mode = 0
    for line in fixationLines:
        if mode == 0:
            if line.startswith("Mutations"):
                mode = 1
        elif mode == 1:
            if line.startswith("Done with fixations"):
                break
            else:
                tempId, permId, mutType, pos, selCoeff, domCoeff, originSubpop, originGen, fixationGen = line.strip().split()
                fixations.append((mutType, int(pos)))
    return fixations

def addMutationsAndGenomesFromSample(sampleText, locs, genomes, sampleSize1):
    mode = 0
    idMapping, mutTypes, tempIdToPos = {}, {}, {}
    introgressedAlleles = []
    fixationLines = []
    for line in sampleText:
        #sys.stderr.write(line+"\n")
        if mode == 0:
            if line.startswith("Emitting fixations"):
                mode = 1
        elif mode == 1:
            fixationLines.append(line.strip())
            if line.startswith("Done with fixations"):
                fixations = parseFixations(fixationLines)
                mode = 2
        elif mode == 2:
            if line.startswith("Mutations"):
                mode = 3
        elif mode == 3:
            if line.startswith("Genomes"):
                mode = 4
            else:
                tempId, permId, mutType, pos, selCoeff, domCoeff, originSubpop, originGen, numCopies = line.strip().split()
                pos, originSubpop, originGen = int(pos), int(originSubpop.lstrip("p")), int(originGen)
                if mutType == "m4":
                    mutType = "m3"
                
                if mutType == "m3":
                    if not pos in locs:
                        locs[pos] = {}
                    locs[pos][permId] = 1
                elif mutType in ["m1","m2"]:
                    tempIdToPos[tempId]=pos
                else:
                    print(tempId, permId, mutType, pos, selCoeff, domCoeff, originSubpop, originGen, numCopies)
                    sys.exit("Encountered mutation type other than m1, m2, or m3. ARRRRGGGGHHHHHHH!!!!!!\n")
                idMapping[tempId]=permId
                mutTypes[tempId]=mutType

        elif mode == 4:
            line = line.strip().split()
            gId, auto = line[:2]
            mutLs = line[2:]
            
            introgressedAlleles.append([])
            for tempId in mutLs:
                if mutTypes[tempId] == "m1" and len(genomes) >= sampleSize1:
                    introgressedAlleles[-1].append(tempIdToPos[tempId])
                elif mutTypes[tempId] == "m2" and len(genomes) < sampleSize1:
                    introgressedAlleles[-1].append(tempIdToPos[tempId])
            for fixationType, fixationPos in fixations:
                if fixationType == "m1" and len(genomes) >= sampleSize1:
                    introgressedAlleles[-1].append(fixationPos)
                elif fixationType == "m2" and len(genomes) < sampleSize1:
                    introgressedAlleles[-1].append(fixationPos)

            genomes.append(set([idMapping[x] for x in mutLs if mutTypes[x] == "m3"]))
    return introgressedAlleles

def readSampleOutFromSlimRun(output, numSamples, sampleSize1):
    totSampleCount = 0
    for line in output.decode("utf-8").split("\n"):
        if line.startswith("Sampling at generation"):
            totSampleCount += 1
    samplesToSkip = totSampleCount-numSamples
    #sys.stderr.write("found {} samples and need to skip {}\n".format(totSampleCount, samplesToSkip))

    if numSamples != 1:
        sys.exit("numSamples={} in 'readSampleOutFromSlimRun' but time series not currently supported. AARRRGGGHHHHH!!!!!!".format(numSamples))

    mode = 0
    samplesSeen = 0
    locs = {}
    genomes = []
    for line in output.decode("utf-8").split("\n"):
        if mode == 0:
            if line.startswith("migProb") or line.startswith("migTime"):
                sys.stderr.write(line+"\n")
            if line.startswith("splitTime"):
                splitTime = int(line.strip().split("splitTime: ")[1])
            if line.startswith("migTime"):
                migTime = int(line.strip().split("migTime: ")[1])
            if line.startswith("Sampling at generation"):
                samplesSeen += 1
                if samplesSeen >= samplesToSkip+1:
                    sampleText = []
                    mode = 1
        elif mode == 1:
            if line.startswith("Done emitting sample"):
                mode = 0
                introgressedAlleles = addMutationsAndGenomesFromSample(sampleText, locs, genomes, sampleSize1)
            else:
                sampleText.append(line)
        if "SEGREGATING" in line:
            sys.stderr.write(line+"\n")
    return locs, genomes, introgressedAlleles

def buildMutationPosMapping(mutLocs, physLen):
    mutMapping = []
    mutLocs.sort()
    for i in range(len(mutLocs)):
        pos, mutId = mutLocs[i]
        mutMapping.append((i, pos, pos/physLen, mutId))
    return mutMapping

def getFreq(mutId, genomes):
    mutCount = 0
    for genome in genomes:
        if mutId in genome:
            mutCount += 1
    return mutCount

def removeMonomorphic(allMuts, genomes):
    newMuts = []
    newLocI = 0
    for locI, loc, contLoc, mutId in allMuts:
        freq = getFreq(mutId, genomes)
        if freq > 0 and freq < len(genomes):
            newMuts.append((newLocI, loc, contLoc, mutId))
            newLocI += 1
    return newMuts

def buildPositionsList(muts, discrete=True):
    positions = []
    for locationIndex, locationDiscrete, locationContinuous, mutId in muts:
        if discrete:
            positions.append(locationDiscrete)
        else:
            positions.append(locationContinuous)
    return positions

def emitMsEntry(positions, segsites, haps, numReps, isFirst=True):
    if isFirst:
        print("slim {} {}".format(len(haps), numReps))
        print("blah")
    print("\n//")
    print("segsites: {}".format(segsites))
    print("positions: "+ " ".join([str(x) for x in positions]))
    for hap in haps:
        print("".join([str(x) for x in hap]))

def processedIntrogressedAlleles(ls):
    ls.sort()
    if len(ls) == 0:
        return []
    else:
        runs = []
        runStart = ls[0]
        for i in range(1, len(ls)):
            if ls[i] > ls[i-1]+1:
                runEnd = ls[i-1]
                runs.append((runStart, runEnd))
                runStart = ls[i]
        runs.append((runStart, ls[-1]))
    return runs

def writeIntrogressedAlleles(repIndex, introgressedAlleles, physLen, outF):
    outF.write("Begin introgressed alleles for rep {}\n".format(repIndex))
    for i in range(len(introgressedAlleles)):
        outF.write("genome {}: {}\n".format(i, ",".join(["%s-%s"%x for x in processedIntrogressedAlleles(introgressedAlleles[i])])))
    outF.write("End rep {}\n".format(repIndex))

if __name__ == '__main__': 
    scriptName, numReps, physLen, donorPop, introgLogFileName, nPerPop, splitTimeCoefficient, migrationTimeCoefficient, migrationProbability, introS = sys.argv[1:]
    numReps = int(numReps)
    physLen = int(physLen)
    donorPop = int(donorPop)
    splitTimeCoefficient = float(splitTimeCoefficient)
    migrationTimeCoefficient = float(migrationTimeCoefficient)
    migrationProbability = float(migrationProbability)
    
    tol=0.5
    outF=open(introgLogFileName, "wt")

    for repIndex in range(numReps):
        #print(repIndex)
        sys.stderr.write("starting rep {}\n".format(repIndex))
        seed = random.randint(0, 2**32-1)
    
        slimCmd = "SLiM/build/slim -seed {} -d physLen={} -d sampleSizePerSubpop={} -d donorPop={} -d st={} -d mt={} -d mp={} -d introS={} {}".format(seed, physLen, nPerPop, donorPop, splitTimeCoefficient, migrationTimeCoefficient, migrationProbability, introS, scriptName)
        sys.stderr.write(slimCmd)
    
    
        procOut = subprocess.Popen(slimCmd.split(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        output, err  = procOut.communicate()
        sys.stderr.write("seed: {}\n".format(seed))
        #print(output.decode("utf-8"))
    
        mutations, genomes, introgressedAlleles = readSampleOutFromSlimRun(output, 1, int(nPerPop))
        newMutLocs = []
        for mutPos in mutations:
            if len(mutations[mutPos]) == 1:
                mutId = list(mutations[mutPos].keys())[0]
                newMutLocs.append((mutPos, mutId))
            else:
                #firstPos = mutPos-tol
                #lastPos = mutPos+tol
                #interval = (lastPos-firstPos)/(len(mutations[mutPos])-1)
                #currPos = firstPos
                for mutId in mutations[mutPos]:
                    newMutLocs.append((mutPos, mutId))
                    #newMutLocs.append((currPos, mutId))
                    #currPos += interval
    
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
        if repIndex == 0:
            emitMsEntry(positions, len(polyMuts), haps, numReps, isFirst=True)
        else:
            emitMsEntry(positions, len(polyMuts), haps, numReps, isFirst=False)
    
        writeIntrogressedAlleles(repIndex, introgressedAlleles, physLen, outF)
    
    outF.close()
