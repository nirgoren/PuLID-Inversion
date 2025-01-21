#!/bin/bash

#SBATCH --job-name=pulid-inversion-debug
#SBATCH --output=array_outputs/pulid-inversion-debug%a.out  # Redirect stdout
#SBATCH --error=array_outputs/pulid-inversion-debug%a.err   # Redirect stderr
#SBATCH --partition=killable
#SBATCH --account=gpu-research
#SBATCH --exclude=rack-omerl-g01,rack-bgw-dgx1,rack-gww-dgx1,n-[202-205]# ,n-[501-503],n-[301-307],n-[350-351]
#SBATCH --time=60  # Max time (minutes)

#SBATCH --ntasks=1  # Number of processes
#SBATCH --gpus=1    # GPUs in total

run_config=configs/run_configs/pulid/rf_pulid_0.5.yaml
data_config=configs/data_configs/people/5.yaml
output_path=results/debug/5

python flux_run_pulid.py --run_yaml $run_config --data_yaml $data_config --output_path $output_path