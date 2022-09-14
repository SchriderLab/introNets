# introNets
Repository for replicating the work in "Detecting introgression at SNP-resolution via
U-Nets and seriation".  In this repo, we provide Python routines to create simulated alignments via MS and SLiM, sort and match populations within alignments, and train a neural network to segment introgressed alleles in either population.

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

Note that we pass the population sizes for the simulations as well as the shape we'd like our formatted input variables to be.
We can now train our model:
```
python3 src/models/train.py --ifile ab.hdf5 --config training_configs/toy_AB.config --odir ab_training --tag iter1
```

The ```--config``` option is passed a config file that we include with our repo that has training settings like batch size and learning rate as well as others:

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
### Simulans vs. Sechelia

Simulating data (1000 replicates per demographic parameter set estimated via DADI):
```
python3 src/data/simulate_msmodified.py --odir /some/where_you_want
python3 src/data/simulate_msmodified.py --odir /some/where_you_want --slurm # if within a SLURM cluster
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
