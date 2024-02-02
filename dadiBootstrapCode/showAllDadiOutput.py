import os
import numpy as np

def readDadiResultsFromDir(dadiResDir, minRuns):
    res = []

    for resFileName in os.listdir(dadiResDir):
        currRes = {}
        mode = 0
        with open(dadiResDir + "/" + resFileName, 'rt') as f:
            for line in f:
                if mode == 0:
                    if line.startswith("After Second Optimization"):
                        mode = 1
                elif mode == 1:
                    if line.strip() == "":
                        mode = 2
                    elif not line.startswith("with u = "):
                        param, val = line.strip().split(":")
                        param = param.strip()
                        val = val.strip()
                        if val != "--":
                            val = float(val.strip())
                            currRes[param] = val
        if 'AIC' in currRes:
            res.append(currRes)

    if len(res) > minRuns:
        res.sort(key=lambda x: x['AIC'])
        return res[0]
    else:
        return None
                
bootResults = []
outDir = "dadiOutput/"

minRuns = 5
for subdir in os.listdir(outDir):
    currRes = readDadiResultsFromDir(outDir + "/" + subdir, minRuns)
    if "all_inter" in subdir:
        if currRes:
            wholeRes = currRes
        else:
            sys.exit("Filed to get {minRuns} successful optimization runs for full data set")
    else:
        if currRes:
            bootResults.append(currRes)

print(f"done reading {len(bootResults)} bootstrap results that had at least {minRuns} successful optimization runs")

print("\t".join([p for p in wholeRes]))
for i in range(len(bootResults)):
    outline = []
    for param in wholeRes:
        outline.append(str(bootResults[i][param]))
    print("\t".join(outline))
