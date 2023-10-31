![alt text](https://github.com/SchriderLab/introNets/blob/main/images/ex_pred.png?raw=true)

# introNets
Repository for replicating the work in "Detecting introgression at SNP-resolution via
U-Nets and seriation".  In this repo, we provide Python routines to create simulated alignments via MS and SLiM, sort and match populations within alignments, and train a neural network to segment introgressed alleles in either population.

For questions or comments on the code email: ddray@email.unc.edu.
I am still updating the documentation and cleaning up the repository to make it more usable.  Stay tuned.

## Dependencies

### General

The project is written entirely in Python 3.x.  In order to build the simulators for the repo you'll need ```gcc``` and ```make``` which for Linux systems are included in the package ```build-essential```.

### SLiM

SLiM is included as a submodule, but needs to be built:

```
cd SLiM/
mkdir build
cd build/
cmake ..
make
```
The scripts expect this binary to be at ```SLiM/build/slim```.

### msmodified

msmodified is again included as a submodule with https://github.com/sriramlab/ArchIE.git.  It can built locally in the folder ArchIE/msmodified if the pre-built binary throws errors: 

```
gcc -o ms ms.c streec.c rand1.c -lm
```

### ms

In order to simulate the cases where no introgression occurs in our test cases or using similar routines, you'll need to compile ms like the following: 

```
gcc -O3 -o ms ms.c streec.c rand2.c -lm
```

Using the ```rand2.c``` is necessary to avoid segfaults with our simulation commands for no introgression. 

## Tutorial

### A toy example
The first example we give in the paper is simple two-population demographic model.

Simulating data (make 1000 examples locally):
```
python3 src/data/simulate_slim.py --direction ab --odir sims/ab --n_jobs 10 --n_replicates 100
python3 src/data/simulate_slim.py --direction ba --odir sims/ab --n_jobs 10 --n_replicates 100
python3 src/data/simulate_slim.py --direction bi --odir sims/bi --n_jobs 10 --n_replicates 100
```

Adding in the arg "--slurm" would submit the simulation commands to SLURM through ```sbatch```. This produces three directories that have 3 * n_jobs files in them.  Each 'job' (locally just sequential simulation runs or via SLURM sbatch submissions) has 3 gzipped output files associated with it for 'n_replicates' simulations each with a random seed: a *.ms.gz file which contains the bi-allelic genotypes for the simulated individuals as well as their SNP positions in an MS style text format, a *_introg.log.gz that contains the introgressed regions for each individual in each simulation like: 

```
Begin introgressed alleles for rep 0
genome 0: 606-1548,4482-5122,5304-5315,7368-7620
genome 1: 3737-4194,5626-6530,8793-9999
genome 2: 3135-3843,3923-4194,5626-6885
genome 3: 0-57,1201-2626,4482-5422
...
genome 124: 3045-3913,9765-9999
genome 125: 285-461,3682-3970,8191-9999
genome 126: 
genome 127: 3682-3970,7546-7822,8100-9394,9813-9999
End rep 0
```

and finally a *_introg.out file that contains the SLiM commands used as well as the migration probabilities and the migration time:

```
starting rep 0
SLiM/build/slim -seed 3119194772 -d physLen=10000 -d sampleSizePerSubpop=64 -d donorPop=3 -d st=4.0 -d mt=0.25 -d mp=1.0 -d introS=0.0 src/SLiM/introg_bidirectional.slimseed: 3119194772
migTime: 11784
migProb12: 0.205016
migProb21: 0.45554
migProbs: 0.102508, 0.22777
...
```

Then we can format the simulations we just created (seriate and match the population alignments and create an hdf5 database).  For example:
```
mpirun -n 8 python3 src/data/format.py --idir sims/ab --ofile ab.hdf5 --pop_sizes 64,64 --out_shape 2,64,128 --pop 1
```

Optionally, you may specify ```--metric``` which defaults to the cosine distance.  This is the distance metric which is used to sort and match the populations.  See https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html for the possible options.  The linear sum assignment between populations is done w.r. to the same metric. We use the h5py format to chunk the data for faster reading when training or evaluating the inference model.  The script takes an argument ```--chunk_size``` that defaults to 4 windows per chunk.  Making the chunk size larger will decrease the read times, but also decrease the randomness of training batches.  

We can calculate some statistics about the hdf5 file.  You will want to do this to properly set the positive weight in the binary cross entropy function as in many simulation cases the number of positive and negative pixels or introgressed alleles is heavily un-balanced.

```
python3 src/data/get_h5_stats.py --ifile ab.hdf5
```
This prints to the console:
```
info for ab.hdf5
neg / pos ratio: 2.1853107954852296
chunk_size: 4
n_replicates: 9991
```

Note that we pass the population sizes for the simulations as well as the shape we'd like our formatted input variables to be.
We can now train our model, but first we need a config file with the proper training parameters where we will specifgy the positive weight for the loss function: 
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
pos_bce_logits_weight = 2.1853107954852296
# whether to label smooth and the upper bound of the uniform random noise used to do so
label_smooth = True
label_noise = 0.01
```

```
python3 src/models/train.py --ifile ab.hdf5 --config training_configs/toy_AB.config --odir ab_training --tag iter1
```

The training script will save the best weights, according to validation loss, and record the training and validation loss metrics as well as accuracy to a CSV in the output directory ```--odir```.  It randomly splits the dataset into train and validation and also saves the hdf5 keys that make up the validation set as a Python pickle file. 

## Evaluation

You can evaluate the trained network to produce a confusion matrix, example predictions and ground truth plots, the reciever operating characteristic curve and the precision recall curves for a separate set of simulations or the validation set from training:

```
python3 src/models/evaluate.py --ifile toy_bi_eval.hdf5 --weights toy_bi.weights --odir toy_bi_eval --n_classes 2
```

# Replication of results shown in publication:

## Simulation

### Simulans vs. Sechelia

Simulating data (1000 replicates per demographic parameter set estimated via DADI) at kb window size (default = 50000):
```
python3 src/data/simulate_msmodified.py --odir /some/where_you_want --window_size 10000 --direction ba
python3 src/data/simulate_msmodified.py --odir /some/where_you_want --windows_size 10000 --direction ba --slurm # if within a SLURM cluster
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

## Training a discriminator (SLiM example)
### Getting the data
We need to simulate all the cases we want to distinguish:

```
python3 src/data/simulate_slim.py --direction ab --odir sims/ab --n_jobs 200 --n_replicates 100
python3 src/data/simulate_slim.py --direction ba --odir sims/ab --n_jobs 200 --n_replicates 100
python3 src/data/simulate_slim.py --direction bi --odir sims/bi --n_jobs 200 --n_replicates 100
python3 src/data/simulate_slim.py --direction none --odir sims/bi --n_jobs 200 --n_replicates 100
```
And then format each one:

```
mpirun -n 8 python3 src/data/format.py --idir sims/ab --ofile sims/ab.hdf5 --pop_sizes 64,64 --out_shape 2,64,128 --pop 1
mpirun -n 8 python3 src/data/format.py --idir sims/ba --ofile sims/ba.hdf5 --pop_sizes 64,64 --out_shape 2,64,128 --pop 0
mpirun -n 8 python3 src/data/format.py --idir sims/bi --ofile sims/bi.hdf5 --pop_sizes 64,64 --out_shape 2,64,128 --pop -1
mpirun -n 8 python3 src/data/format.py --idir sims/none --ofile sims/none.hdf5 --pop_sizes 64,64 --out_shape 2,64,128 --include_zeros```
```

The "--pop" arg and the "--include_zeros" arg are important here as the program uses these to filter out simulations that do not include the desired introgression case by chance.  

Then we can train a ResNet to distinguish windows:

```
python3 src/models/train_discriminator.py --idir sims --odir toy_training/disc --n_classes 4 --batch_size 32
```


## Formatting your own examples
### Probability calibration

As neural networks are known to be poorly calibrated and often overly optimistic i.e. the posterior probabilites they return are often biased despite having a high accuracy when rounded to 0 or 1 for binary classification, we also implemented a routine to learn Platt scaling coefficients to correct for bias and return probabilities that are more meaningful in the frequentist sense (i.e. a randomly selected pixel from the population that has a posterior probability of ~0.8 would be correct ~80%  and incorrect ~20% of the time).  You can use gradient descent to find the Platt coefficients like:

```
python3 src/models/calibrate_probability.py --weights /work/users/d/d/ddray/toy_bi_training/1e5/test.weights --ifile /work/users/d/d/ddray/toy_bi.hdf5 --keys /work/users/d/d/ddray/toy_bi_training/1e5/test_val_keys.pkl --out_channels 2 --ofile platt.npz
```

### Data preparation
If you have a genome that you wish to segement, and a model that you trained on corresponding simulations, first read the data and save it as a numpy array with one key called "positions" which should contain a 1d array of integers that are the base pair positions of each polymorphism.  There should also be exactly two population matrices that contain the genotypes for each individual at each site as binary values (0 or 1).  This routine currently only supports two-population segmentation and no greater (todo).  You pass the keys for these population matrices to the Python routine to format the data for inference: 

```
mpirun -n 6 python3 src/data/format_npz.py --ifile npzs/simulans_sechellia_Scf_X.phased.npz --ofile X.hdf5 --keys simMatrix,sechMatrix --pop_sizes 20,14 --out_shape 2,32,128
```

### Applying a discriminator (detecting windows with > 0 introgressed pixels and the directionality of those pixels)

```
python3 src/models/apply_disc.py --ifile X.hdf5 --weight dros_training/disc/best.weights --ofile X_disc.npz
```

This gives you a single array in the output NPZ that is of the shape (L, n_classes) where the columns have the posterior probabilities of each class.  The classes are in alpha-numeric order, for instance in this case: ('ab', 'ba', 'bi', 'none'). 

### Segmentation (detection of introgressed SNPs)

```
python3 src/models/apply_seg.py --ifile X.hdf5 --weights dros_training/ab --ofile X.npz --pop 1 --verbose
```

This gives you an NPZ with the upsampled predictions for one or both populations (in this case only pop 1) as well as the indices of the up-sampled population which you can use to downsample back to the original pop size: 

```
Python 3.9.0 (default, Nov 15 2020, 14:28:56) 
[GCC 7.3.0] :: Anaconda, Inc. on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import numpy as np
>>> X = np.load('X.npz')
>>> list(X.keys())
['Y', 'x1i', 'x2i']
>>> X['Y'].shape
(32, 1, 897656)
>>> X['x2i']
array([20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 20, 21, 21,
       33, 26, 30, 24, 27, 28, 26, 33, 33, 33, 33, 31, 26, 21, 30],
      dtype=int32)
>>> X['x2i'].shape
(32,)
```

Using these two NPZ files you can now accomplish downstream analysis on the regions predicted to be introgressed!

### Other notes

SLURM environment (UNC CH's Longleaf cluster):

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
