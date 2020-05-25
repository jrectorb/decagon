#!/bin/bash
#SBATCH --account=rrg-bengioy-ad_gpu        # Yoshua pays for your job
#SBATCH --cpus-per-task=10                  # Ask for 10 CPUs
#SBATCH --gres=gpu:1                        # Ask for 1 GPU
#SBATCH --mem=64G                           # Ask for 64 GB of RAM
#SBATCH --time=120:00:00                    # The job will run for 24 hours (has 9:00:00)
#SBATCH -o /home/mkkr/scratch/slurm-%j.out  # Write the log in $SCRATCH

source /home/mkkr/anaconda3/envs/decagon/bin/activate
export PYTHONPATH="${PYTHONPATH}:/home/mkkr/scratch/Projects/11_decagon"
cd /scratch/mkkr/Projects/11_decagon
python -u -m main.main --config configuration-mask-rels.json 1>&2