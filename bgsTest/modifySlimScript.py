import sys

slimFileName = sys.argv[1]

mutRate=5e-9

migRateBlocks = """    N_max = max(N[0,0:(num_populations-1)]);
    defineConstant("splitTime", asInteger(round(burn_in*N_max)));
    splitTimeAgo = 4*N_max;
    defineConstant("simEndTime", splitTime + splitTimeAgo);
    defineConstant("migTimeUpperBound", 0.25);
    rMigTime = rdunif(1, 0, asInteger(round(splitTimeAgo*migTimeUpperBound))); //introgression between 0 gen ago and spltTime*migTimeUpperBound gen ago
    defineConstant("migTime", simEndTime-rMigTime);

    if (donorPop == 1)
    {
        rMigProb = runif(1, 0.1, 0.5);
        cat("migProb: " + rMigProb + "\\n");

        defineConstant("migProb12", rMigProb);
        defineConstant("migProb21", 0);
    }
    else if (donorPop == 2)
    {
        rMigProb = runif(1, 0.1, 0.5);
        defineConstant("migProb12", 0);
        defineConstant("migProb21", rMigProb);
    }
    else if (donorPop == 3)
    {
        rMigProb12 = runif(1, 0.1, 0.5);
        rMigProb21 = runif(1, 0.1, 0.5);
        cat("migProb12: " + rMigProb12 + "\\n");
        cat("migProb21: " + rMigProb21 + "\\n");

        defineConstant("migProb12", rMigProb12/2);
        defineConstant("migProb21", rMigProb21/2);
    }
    else
    {
        defineConstant("migProb12", 0);
        defineConstant("migProb21", 0);
    }
    cat("migProbs: " + migProb12 + ", " + migProb21 + "\\n");"""

migAndSampBlocks="""s1 2000
early() {
     p1.genomes.addNewMutation(m1, 0.0, 0:(physLen-1));
     p2.genomes.addNewMutation(m2, 0.0, 0:(physLen-1));
     p1.setMigrationRates(p2, migProb21);
     p2.setMigrationRates(p1, migProb12);
}

s2 2000
early() {
     p1.setMigrationRates(p2, 0);
     p2.setMigrationRates(p1, 0);
}

s3 3000 late()
{
     cat("Sampling at generation " + sim.cycle + "\\n");
     cat("Emitting fixations\\n");
     sim.outputFixedMutations();
     cat("Done with fixations\\n");
     pop1SampGenomes = sample(p1.genomes, sampleSizePerSubpop1);
     pop2SampGenomes = sample(p2.genomes, sampleSizePerSubpop2);
     fullSamp = c(pop1SampGenomes, pop2SampGenomes);
     fullSamp.output();
     cat("Done emitting sample\\n");
     sim.simulationFinished();
}

1 early() {
     // save this run's identifier, used to save and restore
     defineConstant("simID", getSeed());

     //schedule our events
     community.rescheduleScriptBlock(s1, start=migTime, end=migTime);
     community.rescheduleScriptBlock(s2, start=migTime+1, end=migTime+1);
     community.rescheduleScriptBlock(s3, start=simEndTime, end=simEndTime);
}"""

newLines = []
with open(slimFileName) as f:
    mode = 0
    for line in f:
        if mode == 0:
            if "defineConstant(\"Q\"" in line:
                newLines += [line.rstrip(),""]
                newLines.append("    initializeMutationType(\"m1\", 0.5, \"f\", 0.0);// introduced mutation")
                newLines.append("    initializeMutationType(\"m2\", 0.5, \"f\", 0.0);// introduced mutation")
            elif "defineConstant(\"trees_file\"" in line: #doing this might break stuff but we'll give it a shot
                pass
            elif "initializeMutationType(0, 0.5, \"f\", Q * 0);" in line: #edit our neuts to be m3
                newLines.append("    initializeMutationType(\"m3\", 0.5, \"f\", Q * 0);")
            elif "initializeMutationType(1, 0.5, \"f\", Q * 0);" in line: #skip our beneficials
                pass
            elif "initializeMutationType(2, 0.5, \"g\", Q" in line:
                line = line.replace("MutationType(2", "MutationType(\"m4\"") #edit our deleterious muts to be m4
                newLines.append(line.rstrip())
            elif "initializeMutationRate" in line:
                newLines.append(f"    initializeMutationRate(Q*{mutRate});")
                mode = 1
            elif "initializeGenomicElementType(0, c(0), c(1.0));" in line:
                newLines.append("    initializeGenomicElementType(0, c(m3), c(1.0));")
            elif "initializeGenomicElementType(1, c(1, 2), c(0.0, 0.74));" in line:
                newLines.append("    initializeGenomicElementType(1, c(m3, m4), c(0.26, 0.74));")
            elif "defineConstant(\"N\", asInteger(_N/Q));" in line:
                newLines.append("    defineConstant(\"sampleSizePerSubpop1\", sampling_episodes[1,0]);")
                newLines.append("    defineConstant(\"sampleSizePerSubpop2\", sampling_episodes[1,1]);")
                newLines.append("\n")
                newLines.append(line.rstrip())
                newLines.append("\n")
                newLines.append(migRateBlocks)
            elif "initializeTreeSeq();" in line:
                pass
            elif line.startswith("    // Sample individuals."):
                mode = 2
            else:
                newLines.append(line.rstrip())
        elif mode == 1:
            if ";" in line:
                mode = 0
        elif mode == 2:
            if line.startswith("    }"): #this is the end of our sampling block (in stdpopsim 0.2.0)
                mode = 0
newLines.append(migAndSampBlocks)

for line in newLines:
    print(line)
