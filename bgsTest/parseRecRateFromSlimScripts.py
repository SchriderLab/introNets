import os

def overlap(x1,y1,x2,y2):
    r = 0
    if (y2 < x2 or y1 < x1):
        raise Exception
    elif (y1 <= y2 and y1 >= x2):
        if (x1 > x2):
            r = (x1,y1);
        else:
            r = (x2,y1)
    elif (x1 <= y2 and x1 >= x2):
        if (y1 < y2):
            r = (x1,y1)
        else:
            r = (x1,y2)
    elif (y2 <= y1 and y2 >= x1):
        if (x2 > x1):
            r = (x2,y2)
        else:
            r = (x1,y2)
    elif (x2 <= y1 and x2 >= x1):
        if (y2 < y1):
            r = (x2,y2)
        else:
            r = (x2,y1)
    return r

def sanitizeLine(line):
    line = line.strip().split(",")
    line[-1] = line[-1].rstrip(",")
    line[-1] = line[-1].rstrip(");")
    if line[-1].strip() == "":
        del line[-1]
    return line

def parseRecRatesFromScript(slimFileName):
    with open(slimFileName) as f:
        mode = 0
        for line in f:
            if mode == 0:
                if line.strip().startswith("_recombination_rates"):
                    mode = 1
                    allRates = []
            elif mode == 1:
                rates = sanitizeLine(line)
                #print(rates)
                rates = [float(x) for x in rates]
                allRates.extend(rates)
                if line.strip().endswith(");"):
                    mode = 3
            elif mode == 3:
                if line.strip().startswith("_recombination_ends"):
                    mode = 4
                    allEnds = []
            elif mode == 4:
                ends = sanitizeLine(line)
                #print(ends)
                ends = [float(x) for x in ends]
                allEnds.extend(ends)
                if line.strip().endswith(");"):
                    mode = 5
    return allRates, allEnds

def calcCentralAndTotalRate(rates, ends, centerStart=450000, centerEnd=550000-1):
    start = 0
    ranges = []
    for i in range(len(ends)):
        ranges.append((start, ends[i]))
        start = ends[i]+1

    totL, totRate = 0, 0
    for i in range(len(ranges)):
        s, e = ranges[i]
        l = e-s+1
        totL += l
        totRate += rates[i]*l
    avgRate = totRate / totL

    totalOverLen = 0
    centerRate = 0
    for i in range(len(ranges)):
        s, e = ranges[i]
        overrange = overlap(s, e, centerStart, centerEnd)
        if overrange:
            overLen = overrange[1]-overrange[0]+1
            centerRate += overLen*rates[i]
            totalOverLen += overLen
    assert totalOverLen == centerEnd-centerStart+1
    avgCenterRate = centerRate / totalOverLen

    return avgCenterRate, avgRate

slimScriptPath = "slimScripts/"
slimScriptFileNames = os.listdir(slimScriptPath)

print(f"simFile\tavgRecRateCentral100kb\tavgRecRateFull1Mb")
for slimScriptFileName in slimScriptFileNames:
    rates, ends = parseRecRatesFromScript(slimScriptPath + slimScriptFileName)
    centralRate, totalRate = calcCentralAndTotalRate(rates, ends, centerStart=450000, centerEnd=550000-1)
    print(f"{slimScriptFileName}\t{centralRate}\t{totalRate}")
