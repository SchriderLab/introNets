# introNets
Repository for replicating the work in "Detecting introgression at SNP-resolution via
U-Nets and seriation".  In this repo, we provide Python routines to create simulated alignments via MS and SLiM, sort and match populations within alignments, and train a neural network to segment introgressed alleles in either population.

## Dependencies

### General

The project is written entirely in Python 3.x.  In order to build the simulators for the repo you'll need ```gcc``` and ```make``` which for Linux systems are included in the package ```build-essential```.

### SLiM

SLiM can be installed as follows:

```
git clone https://github.com/MesserLab/SLiM.git
cd SLiM/
mkdir build
cd build/
cmake ..
make
```
The scripts expect this binary to be at ```SLiM/build/slim```.

### msmodified

The msmodified binary is provided by https://github.com/sriramlab/ArchIE.git.  The scripts in this repo expected it to be located in the folder at ```msmodified/ms```.  It can built locally if the pre-built binary throws errors.

## Tutorial

### A toy example
The first example we give in the paper is simple two-population demographic model.

Simulating data (make 1000 examples locally):
```
python3 src/data/simulate_slim.py --direction ab --odir sims/ab --n_jobs 1 --n_replicates 1000 --local
python3 src/data/simulate_slim.py --direction ba --odir sims/ab --n_jobs 1 --n_replicates 1000 --local
python3 src/data/simulate_slim.py --direction bi --odir sims/bi --n_jobs 1 --n_replicates 1000 --local
```

Removing the "--local" arg would submit the simulation commands to SLURM through ```sbatch```. 

Then we can format the simulations we just created (seriate and match the population alignments and create an hdf5 database).  For example:
```
mpirun -n 8 python3 src/data/format.py --idir sims/ab --ofile ab.hdf5 --pop_sizes 64,64 --out_shape 2,128,128
```

We can calculate some statistics about the hdf5 file.  You may want to do this to properly set the positive weight in the binary cross entropy function as in many simulation cases, the number of positive and negative pixels or introgressed alleles may be heavily un-balanced.

```
python3 src/data/get_h5_stats.py --ifile ab.hdf5
```
This prints to the console:
```

```

Note that we pass the population sizes for the simulations as well as the shape we'd like our formatted input variables to be.
We can now train our model:
```
python3 src/models/train.py --ifile ab.hdf5 --config training_configs/toy_AB.config --odir ab_training --tag iter1
```

The training script will save the best weights, according to validation loss, and record the training and validation loss metrics as well as accuracy to a CSV in the output directory ```--odir```.  The ```--config``` option is passed a config file that we include with our repo that has training settings like batch size and learning rate as well as others:

```
[model_params]
# the number of channels in the output image
n_classes = 1 

[training_params]
batch_size = 16
n_epochs = 100
lr = 0.001
# exponential decay of learning rate
schedule = exponential
exp_decay = 0.96
pos_bce_logits_weight = 0.5
# whether to label smooth and the upper bound of the uniform random noise used to do so
label_smooth = True
label_noise = 0.01
```
# Replication of results shown in publication:

## Simulation

### Simulans vs. Sechelia

Simulating data (1000 replicates per demographic parameter set estimated via DADI) at kb window size (default = 50000):
```
python3 src/data/simulate_msmodified.py --odir /some/where_you_want --window_size 10000
python3 src/data/simulate_msmodified.py --odir /some/where_you_want --windows_size 10000 --slurm # if within a SLURM cluster
```

### ArchIE
Simulating data (1000 replicates per job by default = 25000 replicates):
```
python3 src/data/simulate_msmodified.py --model archie --odir /pine/scr/d/d/ddray/archie_sims_new --n_jobs 25
python3 src/data/simulate_msmodified.py --model archie --odir /pine/scr/d/d/ddray/archie_sims_new --slurm --n_jobs 25 # on SLURM
python3 src/data/simulate_msmodified.py --model archie --odir /pine/scr/d/d/ddray/archie_sims_new_wtrees --slurm --n_jobs 25 --trees # with Trees in Newick format. WARNING: trees are very large as MS outputs them in text format and even though they are gzipped this may take up large amounts of disk space.
```

## Formatting

### Simulans vs. Sechelia
```
mpirun python3 src/data/format.py --idir dros_sims_ba/ --ofile dros_ba_cosine.hdf5 --pop_sizes 20,14 --out_shape 2,32,128 --sorting seriate_match --metric cosine
# SLURM example:
sbatch -n 24 --mem=64G -t 2-00:00:00 --wrap "mpirun python3 src/data/format.py --idir dros_sims_new_BA --ofile dros_ba_cityblock.hdf5 --pop_sizes 20,14 --out_shape 2,32,128 --sorting seriate_match --metric cityblock"
```

### ArchIE
```
mpirun python3 src/data/format.py --idir archie_sims --ofile archie_euclidean.hdf5 --pop_sizes 100,100 --out_shape 2,112,128 --sorting seriate_match --metric euclidean
# SLURM example:
sbatch -n 24 --mem=64G -t 2-00:00:00 --wrap "mpirun python3 src/data/format.py --idir archie_sims --ofile archie_euclidean.hdf5 --pop_sizes 100,100 --out_shape 2,112,128 --sorting seriate_match --metric euclidean"
```

## Training

### Simulans vs. Sechelia
```
python3 src/models/train.py --config training_configs/dros_ab.config --ifile dros_ba_cosine.hdf5 --odir training_results/dros_ab_i1
```

### ArchIE
```
python3 src/models/train.py --config training_configs/archie.config --ifile archie_euclidean.hdf5 --odir training_results/archie_i1
```


### Other notes

LL environment:

```
[ddray@longleaf-login5 ~]$ module list

Currently Loaded Modules:
  1) ffmpeg/3.4     3) r/3.3.1        5) openmpi_3.0.0/gcc_6.3.0   7) python/3.7.9
  2) cmake/3.18.0   4) matlab/2019a   6) gcc/6.3.0
  
# for access to the openmpi packages:
module add ~markreed/modules_longleaf # add this to ~/.bashrc if you want it all the time
module add openmpi_3.0.0/gcc_6.3.0
module save # this should be sufficient tho

# install mpi4py locally to LL
pip install --user mpi4py
```
