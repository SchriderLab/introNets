# introNets
Repo to test UNets' ability for inference of introgressed alleles from population genetic alignments.

Formatting simulated data (no sorting as of yet):
```
# spread over 24 cores
sbatch -n 24 --mem=32G -t 2-00:00:00 --wrap "mpirun python3 src/data/format_unet_data.py --verbose --idir /proj/dschridelab/rrlove/ag1000g/data/ms/ms_modified/training/output --ofile /pine/scr/d/d/ddray/unet_data_v1.hdf5"
```

Training:
```
# Volta
sbatch --job-name=pytorch-ac --ntasks=1 --cpus-per-task=1 --mem=32G --time=3-00:00:00 --partition=volta-gpu --gres=gpu:1 --qos=gpu_access --wrap "python3 src/models/train_unet.py --ifile /pine/scr/d/d/ddray/unet_data_v1.hdf5 --odir training_output --tag iter2_test --batch_size 32"
```

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
