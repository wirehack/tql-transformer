#!/usr/bin/bash
#SBATCH --mem=80000                   # Job memory request
#SBATCH --time=0                      # Time limit hrs:min:sec
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4            # Number of CPU cores per task
#SBATCH --output=../log/train.log   # Standard output and error log

module load cuda-91
source activate py37

python ../src/train.py
