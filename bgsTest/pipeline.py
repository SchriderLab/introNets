import os
import runCmdAsJob
import stdpopsim #just to make sure we are in an env with it

baseDir = ""
slimScriptDir = baseDir + "slimScripts"
rawSlimScriptDir = baseDir + "rawSlimScripts"
outDir = baseDir + "slimOutput"
logDir = baseDir + "slimLogs"

os.system(f"mkdir -p {rawSlimScriptDir} {slimScriptDir} {outDir} {logDir}")

jobName = "bgsIntroSlim"
launchFileName = "bgsIntroSlim.sh"
wallTime = "12:00:00"
qName = "general"
mem = "64G"

numReps=100

for i in range(numReps):
    slimScriptFileName = f"{slimScriptDir}/bgsSim_{i}.slim"
    slimScriptFileName = f"{rawSlimScriptDir}/bgsSim_{i}.slim"
    outFileName = f"{outDir}/bgsSim_{i}.out"
    logFileName = f"{logDir}/bgsSim_{i}.log"

    cmd = f"python extractSlimScriptForDmelAnnotDFE_randomWindow.py {rawSlimScriptDir}/bgsSim_{i}.slim\n"
    cmd += f"python modifySlimScript.py {rawSlimScriptDir}/bgsSim_{i}.slim > {slimScriptDir}/bgsSim_{i}.slim\n"
    cmd += f"slim -d physLen=1000000 -d donorPop=3 {slimScriptDir}/bgsSim_{i}.slim | gzip > {outDir}/bgsSim_{i}.out.gz"
    runCmdAsJob.runCmdAsJobWithoutWaitingWithLog(cmd, jobName, launchFileName, wallTime, qName, mem, logFileName)
