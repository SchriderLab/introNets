# -*- coding: utf-8 -*-
import os
import argparse
import logging

SIZE_A = 20
SIZE_B = 14
N_SITES = 10000

# use this format to tell the parsers
# where to insert certain parts of the script
# ${imports}

def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser()
    # my args
    parser.add_argument("--verbose", action = "store_true", help = "display messages")
    parser.add_argument("--ifile", default = "None")

    parser.add_argument("--idir_a", default = "None")
    parser.add_argument("--idir_b", default = "None")
    parser.add_argument("--weights", default = "None")

    parser.add_argument("--mlp_boost", action = "store_true")
    parser.add_argument("--n_steps", default = "1")

    # ${args}
    parser.add_argument("--odir", default = "None")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
        logging.debug("running in verbose mode")
    else:
        logging.basicConfig(level=logging.INFO)

    if args.odir != "None":
        if not os.path.exists(args.odir):
            os.system('mkdir -p {}'.format(args.odir))
            logging.debug('root: made output directory {0}'.format(args.odir))
    # ${odir_del_block}

    return args

def main():
    args = parse_args()

    idir_b = args.idir_b
    weights = args.weights
    job_id = None
    mlp_weights = "None"

    # ${code_blocks}
    for ix in range(int(args.n_steps)):
        # get the next generation of parameters, and generate
        # a text file of the simulation commands
        ##### Parameter estimation for next generation
        ##### ------------------------------------------>
        odir =  os.path.join(args.odir, 'gen{0:03d}'.format(ix + 1))
        os.system('mkdir -p {}'.format(odir))

        cmd = 'python3 src/models/pytorch/adversarial_evolution_mlp.py --ifile {0} --odir {1} --weights {2}'.format(args.ifile, os.path.join(args.odir, 'gen{0:03d}'.format(ix + 1)), mlp_weights)
        mlp_weights = os.path.join(odir, 'regression.weights')

        if job_id is None:
            sbatch_cmd = 'sbatch -t 1-00:00:00 --mem=16G --wrap "{}"'.format(cmd)
        else:
            sbatch_cmd = 'sbatch -t 1-00:00:00 --mem=16G --depend=afterok:{1} --wrap "{0}"'.format(cmd, job_id)

        print('submitting parameter evolution...:')
        print(sbatch_cmd)

        job_id = os.popen(sbatch_cmd).read().split(" ")[-1].strip('\n')
        print('array jobid is {}'.format(job_id))

        ##### Simulation array job
        #### ------------------------>
        N = 1821
        fname = os.path.join(odir, 'cmds.txt')
        f = open(fname, 'w')

        for ix in range(N):
            odir_ = os.path.join(odir, '{0:04d}'.format(ix))
            cmd = "cd %s; %s %d %d -t tbs -r tbs %d -I 2 %d %d -n 1 tbs -n 2 tbs -eg 0 1 6.576808 -eg 0 2 -7.841388 -ma x tbs tbs x -ej tbs 2 1 -en tbs 1 1 -es tbs 2 tbs -ej tbs 3 1 < %s > mig.msOut" % (odir_, os.path.join(os.getcwd(), 'msmodified/ms'), SIZE_A + SIZE_B, 10, N_SITES, SIZE_A, SIZE_B, 'mig.tbs')
            f.write(cmd + '\n')

        f.close()
        sbatch_cmd = 'sbatch --array=0-{0} -o {3} --depend=afterany:{2} -t 05:00:00 --mem=16G src/SLURM/array.sh {1}'.format(N, fname, job_id, os.path.join(odir, 'sim_%A_%a.out'))

        job_id = os.popen(sbatch_cmd).read().split(" ")[-1].strip('\n')
        print('array jobid is {}'.format(job_id))

        #### Formatting
        ### -------------->
        cmd = 'python3 src/data/rewrite_data_dros.py --idir {0} --odir {1} --idir_other {2} --direction mig'.format(odir, os.path.join(odir, 'npzs'), idir_b)
        idir_b = os.path.join(odir, 'npzs')

        sbatch_cmd = 'sbatch -t 1-00:00:00 --mem=4G --depend=afterok:{1} --wrap "{0}"'.format(cmd, job_id)

        job_id = os.popen(sbatch_cmd).read().split(" ")[-1].strip('\n')
        print('array jobid is {}'.format(job_id))

        cmd = 'python3 src/models/pytorch/train_discriminator.py --idir_a {0} --idir_b {1} --weights {2} --odir {3} --ofile {4} --tag discriminator'.format(idir_b, args.idir_a, weights, odir, args.ifile)
        sbatch_cmd = 'sbatch --job-name=pytorch-ac --ntasks=1 --cpus-per-task=1 --mem=32G --time=3-00:00:00 --partition=volta-gpu --gres=gpu:1 --qos=gpu_access --depend=afterok:{1} --wrap "{0}"'.format(cmd, job_id)
        weights = os.path.join(odir, 'discriminator.weights')

        job_id = os.popen(sbatch_cmd).read().split(" ")[-1].strip('\n')
        print('array jobid is {}'.format(job_id))




if __name__ == '__main__':
    main()
