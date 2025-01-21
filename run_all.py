from pathlib import Path
import subprocess

# Directories
run_yaml_dir = Path("configs/run_configs/portraits")
data_yaml_dir = Path("configs/data_configs/portraits")
output_base_dir = Path("results/portraits")

# Get all YAML files in the specified directories (recursive search)
run_yaml_files = list(run_yaml_dir.rglob("*.yaml"))
data_yaml_files = list(data_yaml_dir.rglob("*.yaml"))

# Iterate through all combinations of run_yaml and data_yaml
for run_yaml_path in run_yaml_files:
    run_yaml_name = run_yaml_path.stem  # Filename without extension

    for data_yaml_path in data_yaml_files:
        data_yaml_name = data_yaml_path.stem  # Filename without extension

        # Define the output path based on the combination
        output_path = output_base_dir / run_yaml_name / data_yaml_name
        output_path.mkdir(parents=True, exist_ok=True)

        # Command to run the script
        command = [
            "python", "flux_run_pulid.py",
            "--run_yaml", str(run_yaml_path),
            "--data_yaml", str(data_yaml_path),
            "--output_path", str(output_path)
        ]

        # Execute the command
        print(f"Running: {' '.join(command)}")
        subprocess.run(command)