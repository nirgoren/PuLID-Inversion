#!/bin/bash

#SBATCH --job-name=pulid-inversion
#SBATCH --output=array_outputs/pulid-inversion%a.out  # Redirect stdout
#SBATCH --error=array_outputs/pulid-inversion%a.err   # Redirect stderr
#SBATCH --partition=killable
#SBATCH --account=gpu-research
#SBATCH --exclude=rack-omerl-g01,rack-bgw-dgx1,rack-gww-dgx1,n-[202-205] # ,n-[501-503],n-[301-307],n-[350-351]
#SBATCH --time=60  # Max time (minutes)

#SBATCH --ntasks=1  # Number of processes
#SBATCH --gpus=1    # GPUs in total
#SBATCH --array=1-4 # Array range (one task per job)

data_config_dir=configs/data_configs/people/
base_output_dir=results/

if [ "$#" -eq 2 ]; then
    data_config_dir=$1
    base_output_dir=$2
fi
# Define tasks for each job
case $SLURM_ARRAY_TASK_ID in
    1)
        python run_all.py --run_yaml_dir configs/run_configs/rf/rf_soft_guidance.yaml --data_yaml_dir $data_config_dir --base_output_dir $base_output_dir
        ;;
    2)
        python run_all.py --run_yaml_dir configs/run_configs/pulid/rf_pulid_0.5.yaml --data_yaml_dir $data_config_dir --base_output_dir $base_output_dir
        ;;
    3)
        python run_all.py --run_yaml_dir configs/run_configs/pulid/rf_pulid_0.7.yaml --data_yaml_dir $data_config_dir --base_output_dir $base_output_dir
        ;;
    4)
        python run_all_data.py --run_config_path configs/run_configs/ddim/ddim_ipa_scale_0.4.yaml --data_config_dir $data_config_dir --base_output_dir $base_output_dir --override_config_path $override_config_path
        ;;
    *)
        echo "Invalid SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
        exit 1
        ;;
esac