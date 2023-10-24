import sys, os, random
import stdpopsim
from contextlib import redirect_stdout

Q=1000
speciesName = "DroMel"
sampleSize=64
sampleSizes = [sampleSize*2,sampleSize*2]
defaultModel = "const2pop"
targetPops = [0,1]
contigName = "chr3L"
geneticMapName = "ComeronCrossoverV2_dm6"
dfeName = "Gamma_H17"
annotName = "FlyBase_BDGP6.32.51_exons"
scriptFileName = sys.argv[1]

engine = stdpopsim.get_engine('slim')
species = stdpopsim.get_species(speciesName)

N = int(1e6)
mu=5e-9
r=5e-9
winLen = int(1e6)

#split 4N gen ago
#daughter pops both have N=1e6
model = stdpopsim.IsolationWithMigration(NA=N, N1=N, N2=N, T=4*N, M12=0, M21=0)

samples = model.get_samples(*sampleSizes)

contig = species.get_contig(contigName, genetic_map=geneticMapName)
chrLen = contig.length
winStart = random.randint(1, chrLen-winLen+1)
winEnd = winStart+winLen
contig = species.get_contig(contigName, genetic_map=geneticMapName, left=winStart, right=winEnd)

dfe = species.get_dfe(dfeName)
exons = species.get_annotations(annotName)
exon_intervals = exons.get_chromosome_annotations(contigName)
contig.add_dfe(intervals=exon_intervals, DFE=dfe)

with open(scriptFileName, 'w') as f:
    with redirect_stdout(f):
        ts = engine.simulate(model, contig, samples, slim_script=True, verbosity=2, slim_scaling_factor=Q, slim_burn_in=20)
