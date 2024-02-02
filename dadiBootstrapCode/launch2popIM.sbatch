#!/bin/bash
#SBATCH --job-name=dadiOpt
#SBATCH --time=48:00:00
#SBATCH --partition=general
#SBATCH --output=dadi_jobs.log
#SBATCH --mem=16G
#SBATCH --requeue
#SBATCH -n 8
#SBATCH -N 1
#SBATCH --export=ALL

source activate python27
currHost=$(hostname)
baseDir=/nas/longleaf/home/dschride/projects/introgression/simSech/dadiStuff
inFile=$baseDir/dadiIn/sample_${SLURM_ARRAY_TASK_ID}.sfs
mkdir -p $baseDir/dadiPlots/2popIM_${SLURM_ARRAY_TASK_ID} $baseDir/dadiOutput/2popIM_${SLURM_ARRAY_TASK_ID} $baseDir/dadiLogs/2popIM_${SLURM_ARRAY_TASK_ID}

cd $baseDir/dadiLogs/2popIM_${SLURM_ARRAY_TASK_ID}

for i in {0..9}
do
    echo "run $i"
    mpiexec -n $SLURM_NTASKS -host $currHost python $baseDir/2popIM.py $inFile $SLURM_NTASKS $baseDir/dadiPlots/2popIM_${SLURM_ARRAY_TASK_ID}/run_${i}.pdf 1> $baseDir/dadiOutput/2popIM_${SLURM_ARRAY_TASK_ID}/run_${i}.txt 2> run_${i}.log
done
