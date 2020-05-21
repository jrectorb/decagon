#!/bin/bash
#SBATCH --account=rrg-bengioy-ad_cpu                  # Yoshua pays for your job
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G                               # Ask for 8 GB of RAM
#SBATCH --time=0:30:00                        # The job will run for 24 hours
#SBATCH -o /home/mkkr/scratch/slurm-%j.out  # Write the log in $SCRATCH

export PYTHONPATH="${PYTHONPATH}:/home/mkkr/scratch/Projects/8_affinity"

cd /home/mkkr/scratch/Projects/9_decagon
python test.py 1>&2

