#! /usr/bin/bash

#---------------------------------------------------------------------------------
# Account information

#SBATCH --account=pi-naragam              # basic (default), phd, faculty, pi-<account>

#---------------------------------------------------------------------------------
# Directory

#SBATCH --output=./output/slurm/slurm-%j.out

#---------------------------------------------------------------------------------
# Resources requested

#SBATCH --partition=gpu_h100       # standard (default), long, gpu, mpi, highmems
#SBATCH --mem=32G           # requested memory
#SBATCH --gpus-per-node=1
#SBATCH --time=0:30:00          # wall clock limit (d-hh:mm:ss)

#---------------------------------------------------------------------------------
# Job specific name (helps organize and track progress of jobs)

#SBATCH --job-name=MNIST_conv    # user-defined job name

#---------------------------------------------------------------------------------
dataset=MNIST
model="ShallowConvVAE"
latent_dim=2
beta=1e-4
epochs=20
output_dir=./output/$dataset/$model/$SLURM_JOB_ID

echo "Job ID: $SLURM_JOB_ID"
echo "Job User: $SLURM_JOB_USER"
echo "Num Cores: $SLURM_JOB_CPUS_PER_NODE"
echo
echo "Dataset: $dataset"
echo "Model: $model"
echo "Latent Dimension: $latent_dim"
echo "Beta: $beta"
echo "Epochs: $epochs"
echo "Output Directory: $output_dir"

#---------------------------------------------------------------------------------
# Load modules
module load python/booth/3.12

#---------------------------------------------------------------------------------
# Running script

mkdir -p $output_dir
python3 training.py $output_dir $dataset $model $latent_dim $beta $epochs
