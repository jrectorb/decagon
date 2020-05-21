#!/bin/bash
#SBATCH --account=rrg-bengioy-ad_gpu        # Yoshua pays for your job
#SBATCH --cpus-per-task=10                  # Ask for 10 CPUs
#SBATCH --gres=gpu:4                        # Ask for 3 GPU
#SBATCH --mem=76G                           # Ask for 76 GB of RAM
#SBATCH --time=36:00:00                     # The job will run for 24 hours (has 9:00:00)
#SBATCH -o /home/mkkr/scratch/slurm-%j.out  # Write the log in $SCRATCH

source /home/mkkr/anaconda3/envs/decagon/bin/activate
export PYTHONPATH="${PYTHONPATH}:/home/mkkr/scratch/Projects/11_decagon"
cd /scratch/mkkr/Projects/11_decagon
python -u -m main.main 1>&2
