#!/bin/bash
#SBATCH --job-name=D_test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=12GB
#SBATCH --time=167:59:59
#SBATCH --gres=gpu
#SBATCH --output=slurm_test.out


# clear all modules
module purge

# call singularity env
singularity exec --nv \
            --overlay /scratch/hq443/singularity/torch.ext3:ro \
            /scratch/work/public/singularity/cuda11.2.2-cudnn8-devel-ubuntu20.04.sif \
            /bin/bash -c "source /ext3/env.sh; \
                  python test.py; \
                  exit;"
