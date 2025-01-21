import argparse
from pathlib import Path
import subprocess

def run_all(run_yaml_dir, data_yaml_dir, output_base_dir):
    if run_yaml_dir.is_file():
        run_yaml_files = [run_yaml_dir]
    else:
        # Get all YAML files in the specified directories (recursive search)
        run_yaml_files = list(run_yaml_dir.rglob("*.yaml"))
    if data_yaml_dir.is_file():
        data_yaml_files = [data_yaml_dir]
    else:
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Run all combinations of run and data YAML files")
    parser.add_argument("--run_yaml_dir", type=Path, required=True, help="Directory containing run YAML files")
    parser.add_argument("--data_yaml_dir", type=Path, required=True, help="Directory containing data YAML files")
    parser.add_argument("--output_base_dir", type=Path, required=True, help="Base directory for output")
    args = parser.parse_args()
    run_all(args.run_yaml_dir, args.data_yaml_dir, args.output_base_dir)
    print("All combinations completed")