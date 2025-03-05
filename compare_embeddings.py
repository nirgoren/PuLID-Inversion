import torch
import numpy as np
from pathlib import Path
import argparse

def compare_embeddings(file1: Path, file2: Path, atol: float = 1e-6, rtol: float = 1e-5):
    """
    Compare two saved tensors and return a summary of differences.
    """
    try:
        embeddings1 = torch.load(file1, map_location="cpu", weights_only=True)
        embeddings2 = torch.load(file2, map_location="cpu", weights_only=True)

        if embeddings1.shape != embeddings2.shape:
            return f"Shape mismatch: {embeddings1.shape} vs {embeddings2.shape}"

        embeddings1 = embeddings1.to(dtype=torch.float32)
        embeddings2 = embeddings2.to(dtype=torch.float32)

        if torch.equal(embeddings1, embeddings2):
            return "Exactly equal"

        is_close = torch.allclose(embeddings1, embeddings2, rtol=rtol, atol=atol)
        abs_diff = torch.abs(embeddings1 - embeddings2)
        mean_abs_diff = abs_diff.mean().item()
        max_abs_diff = abs_diff.max().item()
        cos_sim = torch.nn.functional.cosine_similarity(embeddings1.flatten(), embeddings2.flatten(), dim=0).item()

        return (
            f"Shape: {embeddings1.shape}, "
            f"Close: {is_close}, "
            f"Mean Abs Diff: {mean_abs_diff:.6f}, "
            f"Max Abs Diff: {max_abs_diff:.6f}, "
            f"Cosine Sim: {cos_sim:.6f}"
        )
    except Exception as e:
        return f"Error: {str(e)}"

def compare_directories(dir1: str, dir2: str, atol: float = 1e-6, rtol: float = 1e-5):
    """
    Compare all .pth files between two directories and print a summary.
    """
    dir1_path = Path(dir1)
    dir2_path = Path(dir2)

    if not dir1_path.exists() or not dir2_path.exists():
        print(f"Error: One or both directories do not exist: {dir1}, {dir2}")
        return

    # Get list of .pth files in each directory
    files1 = set(f.name for f in dir1_path.glob("*.pth"))
    files2 = set(f.name for f in dir2_path.glob("*.pth"))

    # Find common files
    common_files = files1.intersection(files2)
    if not common_files:
        print("No common .pth files found between directories.")
        return

    print(f"Comparing {len(common_files)} common files between {dir1} and {dir2}:\n")
    print(f"{'File':<30} {'Result':<60}")
    print("-" * 90)

    for filename in sorted(common_files):
        file1 = dir1_path / filename
        file2 = dir2_path / filename
        result = compare_embeddings(file1, file2, atol, rtol)
        print(f"{filename:<30} {result:<60}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare .pth files between two directories.")
    parser.add_argument("--dir1", required=True, help="Path to first directory (e.g., latents/1)")
    parser.add_argument("--dir2", required=True, help="Path to second directory (e.g., latents/2)")
    parser.add_argument("--atol", type=float, default=1e-6, help="Absolute tolerance")
    parser.add_argument("--rtol", type=float, default=1e-5, help="Relative tolerance")
    args = parser.parse_args()

    compare_directories(args.dir1, args.dir2, args.atol, args.rtol)