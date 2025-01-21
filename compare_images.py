from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import argparse

from flux.image_utils import add_label_to_image

def calculate_square_difference(file1, file2):
    """
    Calculate the sum of squared differences (SSD) between two .npy files.
    """
    if not file1.exists() or not file2.exists():
        raise FileNotFoundError(f"Missing file: {file1} or {file2}")
    latents1 = np.load(file1)
    latents2 = np.load(file2)
    return np.sum((latents1 - latents2) ** 2)

def find_sample_paths(results_folder):
    """
    Recursively finds all sample folders in the given results_folder.
    Returns a dictionary where keys are relative paths of samples and values are full paths.
    """
    sample_paths = {}
    for path in results_folder.rglob("results.jpg"):
        relative_sample_path = path.parent.relative_to(results_folder)
        sample_paths[str(relative_sample_path)] = path.parent
    return sample_paths

def stack_images(results_folders, out_folder):
    """
    Stacks the results.jpg files from corresponding sample folders (nested) 
    in the given results_folders and saves the stacked images in out_folder.
    """
    results_folders = [Path(folder) for folder in results_folders]
    out_folder = Path(out_folder) / "stacked_images"
    out_folder.mkdir(parents=True, exist_ok=True)

    # Collect all sample paths from each results folder
    all_sample_paths = {}
    for folder in results_folders:
        sample_paths = find_sample_paths(folder)
        for rel_path, full_path in sample_paths.items():
            all_sample_paths.setdefault(rel_path, []).append(full_path)

    # Stack images for each relative sample path
    for rel_path, paths in all_sample_paths.items():
        images_to_stack = []
        
        for folder_path in paths:
            result_file = folder_path / "results.jpg"
            if result_file.exists():
                image_to_stack = Image.open(result_file)
                # Add a label to the image
                label = str(folder_path)
                image_to_stack = add_label_to_image(image_to_stack, label)
                images_to_stack.append(image_to_stack)
            else:
                print(f"Warning: Missing results.jpg for sample '{rel_path}' in '{folder_path}'")
        
        if images_to_stack:
            widths, heights = zip(*(img.size for img in images_to_stack))
            total_height = sum(heights)
            max_width = max(widths)
            
            stacked_image = Image.new("RGB", (max_width, total_height))
            y_offset = 0
            for img in images_to_stack:
                stacked_image.paste(img, (0, y_offset))
                y_offset += img.size[1]
            
            output_path = out_folder / f"{rel_path.replace('/', '_')}_stacked.jpg"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            stacked_image.save(output_path)
            print(f"Saved stacked image to: {output_path}")

def compare_latents(results_folders, out_folder):
    """
    Compare the latents files for each sample across experiments and create a combined plot.
    """
    results_folders = [Path(folder) for folder in results_folders]
    out_folder = Path(out_folder)
    out_folder.mkdir(parents=True, exist_ok=True)

    # gather all sample folders
    sample_names = []
    for folder in results_folders:
        sample_names.extend([f.name for f in folder.iterdir() if f.is_dir()])
    # remove duplicates
    sample_names = list(set(sample_names))
    sample_names.sort()
    

    sample_results = {}

    for sample_name in sample_names:
        sample_results[sample_name] = []

        for folder in results_folders:
            reconstruction_latents = folder / sample_name / "original_reconstruction/latents/latents_0000.npy"
            inversion_latents = folder / sample_name / "original_inversion/latents/latents_0000.npy"
            
            if reconstruction_latents.exists() and inversion_latents.exists():
                ssd = calculate_square_difference(reconstruction_latents, inversion_latents)
                sample_results[sample_name].append(ssd)
            else:
                print(f"Warning: Missing latents file for sample '{sample_name}' in '{folder}'")
                sample_results[sample_name].append(None)

    plot_combined_latents(sample_results, results_folders, out_folder)

def plot_combined_latents(sample_results, results_folders, out_folder):
    """
    Create a single bar plot for all samples and experiments, showing the SSD comparisons.
    """
    samples = list(sample_results.keys())
    experiments = [folder.name for folder in results_folders]
    bar_width = 0.2
    x = np.arange(len(samples))  # Positions for sample groups

    plt.figure(figsize=(12, 6))
    
    # Plot bars for each experiment
    for i, experiment in enumerate(experiments):
        ssds = [sample_results[sample][i] if sample_results[sample][i] is not None else 0 for sample in samples]
        plt.bar(x + i * bar_width, ssds, width=bar_width, label=experiment, edgecolor='black')

    # Labeling and aesthetics
    plt.title("Comparison of Latent SSDs Across Samples and Experiments")
    plt.xticks(x + bar_width * (len(experiments) - 1) / 2, samples, rotation=45, ha='right')
    plt.xlabel("Samples")
    plt.ylabel("Sum of Squared Differences (SSD)")
    plt.legend(title="Experiments")
    plt.tight_layout()

    # Save the plot
    plot_path = out_folder / "combined_latents_comparison.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved combined latent comparison plot to: {plot_path}")

def main():
    parser = argparse.ArgumentParser(description="Stack results.jpg and compare latent files across experiments.")
    parser.add_argument(
        "results_folders",
        nargs="+",
        help="List of paths to the results folders to compare."
    )
    parser.add_argument(
        "--out_folder",
        required=True,
        help="Path to the output folder where stacked images and plots will be saved."
    )
    args = parser.parse_args()

    # Perform both tasks
    stack_images(args.results_folders, args.out_folder)
    compare_latents(args.results_folders, args.out_folder)

if __name__ == "__main__":
    main()
