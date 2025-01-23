#!/bin/bash

#SBATCH --job-name=pulid-inversion-unsplash_ode
#SBATCH --output=array_outputs/pulid-inversion-unsplash_ode%a.out  # Redirect stdout
#SBATCH --error=array_outputs/pulid-inversion-unsplash_ode%a.err   # Redirect stderr
#SBATCH --partition=killable
#SBATCH --account=gpu-research
#SBATCH --exclude=rack-omerl-g01,rack-bgw-dgx1,rack-gww-dgx1,n-[202-205],n-[301-302],n-[501-503],n-602 # ,n-[350-351]
#SBATCH --time=600  # Max time (minutes)

#SBATCH --ntasks=1  # Number of processes
#SBATCH --gpus=1    # GPUs in total
#SBATCH --array=1,6-11 # Array range (one task per job)

data_config_dir=configs/data_configs/unsplash_yamls
base_output_dir=results/unsplash_ode

if [ "$#" -eq 2 ]; then
    data_config_dir=$1
    base_output_dir=$2
fi
# Define tasks for each job
case $SLURM_ARRAY_TASK_ID in
    1)
        python run_all.py --run_yaml_dir configs/run_configs/ode_inversion/ode_pulid_0.0.yaml --data_yaml_dir $data_config_dir --output_base_dir $base_output_dir
        ;;
    2)
        python run_all.py --run_yaml_dir configs/run_configs/ode_inversion/ode_pulid_0.1.yaml --data_yaml_dir $data_config_dir --output_base_dir $base_output_dir
        ;;
    3)
        python run_all.py --run_yaml_dir configs/run_configs/ode_inversion/ode_pulid_0.2.yaml --data_yaml_dir $data_config_dir --output_base_dir $base_output_dir
        ;;
    4)
        python run_all.py --run_yaml_dir configs/run_configs/ode_inversion/ode_pulid_0.3.yaml --data_yaml_dir $data_config_dir --output_base_dir $base_output_dir
        ;;
    5)
        python run_all.py --run_yaml_dir configs/run_configs/ode_inversion/ode_pulid_0.4.yaml --data_yaml_dir $data_config_dir --output_base_dir $base_output_dir
        ;;
    6)
        python run_all.py --run_yaml_dir configs/run_configs/ode_inversion/ode_pulid_0.5.yaml --data_yaml_dir $data_config_dir --output_base_dir $base_output_dir
        ;;
    7)
        python run_all.py --run_yaml_dir configs/run_configs/ode_inversion/ode_pulid_0.6.yaml --data_yaml_dir $data_config_dir --output_base_dir $base_output_dir
        ;;
    8)
        python run_all.py --run_yaml_dir configs/run_configs/ode_inversion/ode_pulid_0.7.yaml --data_yaml_dir $data_config_dir --output_base_dir $base_output_dir
        ;;
    9)
        python run_all.py --run_yaml_dir configs/run_configs/ode_inversion/ode_pulid_0.8.yaml --data_yaml_dir $data_config_dir --output_base_dir $base_output_dir
        ;;
    10)
        python run_all.py --run_yaml_dir configs/run_configs/ode_inversion/ode_pulid_0.9.yaml --data_yaml_dir $data_config_dir --output_base_dir $base_output_dir
        ;;
    11)
        python run_all.py --run_yaml_dir configs/run_configs/ode_inversion/ode_pulid_1.0.yaml --data_yaml_dir $data_config_dir --output_base_dir $base_output_dir
        ;;
    *)
        echo "Invalid SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
        exit 1
        ;;
esac