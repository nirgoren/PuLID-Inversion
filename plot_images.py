from pathlib import Path
from flux.image_utils import find_and_plot_images

if __name__ == "__main__":
    root_dir = Path("results")
    # find directories with "data_config.yaml"

    data_config_files = list(root_dir.rglob("data_config.yaml"))
    for data_config_file in data_config_files:
        data_dir = data_config_file.parent
        output_file = data_dir / "results.jpg"
        find_and_plot_images(data_dir, output_file, recursive=True)