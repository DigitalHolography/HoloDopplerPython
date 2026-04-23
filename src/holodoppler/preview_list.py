import os
import json
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from math import ceil, sqrt

from holodoppler.Holodoppler import Holodoppler
from matlab_imresize.imresize import imresize


# =========================
# Utility functions
# =========================

def make_square(img):
    """
    Center-crop to square.
    
    Parameters:
        img (np.ndarray): 2D image
        
    Returns:
        np.ndarray: square image
    """
    H, W = img.shape

    return imresize(img, output_shape=(max(H,W),max(H,W)))


def normalize_image(img, mode="percentile"):
    """
    Normalize image for visualization.
    
    mode:
        - "minmax"
        - "percentile" (robust for speckle / outliers)
    """
    if mode == "minmax":
        img = (img - img.min()) / (img.max() - img.min() + 1e-12)
    
    elif mode == "percentile":
        p1, p99 = np.percentile(img, (1, 99))
        img = np.clip((img - p1) / (p99 - p1 + 1e-12), 0, 1)

    return img

# =========================
# Montage builder
# =========================

def build_montage_with_coords(images):
    """
    Build montage and return tile coordinates.
    """
    N = len(images)
    grid_size = ceil(sqrt(N))

    h, w = images[0].shape
    montage = np.zeros((grid_size * h, grid_size * w), dtype=np.float32)

    coords = []

    for idx, img in enumerate(images):
        i = idx // grid_size
        j = idx % grid_size

        y0 = i * h
        x0 = j * w

        montage[y0:y0+h, x0:x0+w] = img
        coords.append((x0, y0))

    return montage, coords, h, w


def build_montage(images):
    """
    Assemble square images into a grid montage.
    """
    N = len(images)
    grid_size = ceil(sqrt(N))

    h, w = images[0].shape
    montage = np.zeros((grid_size * h, grid_size * w), dtype=np.float32)

    for idx, img in enumerate(images):
        i = idx // grid_size
        j = idx % grid_size

        montage[i*h:(i+1)*h, j*w:(j+1)*w] = img

    return montage


# =========================
# Main pipeline
# =========================

def generate_m0_montage(txt_path, output_path, parameters_path):
    
    # --- Load parameters
    with open(parameters_path) as f:
        parameters = json.load(f)

    # --- Read holo file list
    with open(txt_path, "r") as f:
        holo_paths = [line.strip().replace('"', '') for line in f if line.strip()]

    print(f"Found {len(holo_paths)} holo files")

    # --- Init Holodoppler
    HD = Holodoppler(backend="cupyRAM", pipeline_version="latest")

    images = []
    valid_paths = []

    # =========================
    # Processing loop
    # =========================
    for path in tqdm(holo_paths, desc="Processing holo previews"):

        # try:
            HD.load_file(path)

            # Minimal read (adjust if needed)
            _ = HD.read_frames(0, 1)

            res = HD.render_moments(parameters, tictoc=False)

            M0 = res["M0"].get()  # GPU → CPU

            # --- Square enforcement
            M0 = make_square(M0)

            # --- Normalize (for visualization consistency)
            M0 = normalize_image(M0, mode="percentile")

            images.append(M0)
            valid_paths.append(path)

        # except Exception as e:
        #     print(f"[WARNING] Failed on {path}: {e}")

    if len(images) == 0:
        raise RuntimeError("No valid images processed.")

    # =========================
    # Build montage
    # =========================
    montage, coords, h, w = build_montage_with_coords(images)

    # =========================
    # Plot & overlay
    # =========================
    fig, ax = plt.subplots(figsize=(12, 12))

    ax.imshow(montage, cmap="gray")
    ax.axis("off")

    if True:
        fontsize = np.clip(int(0.04 * h),3,8)

        for (x0, y0), path in zip(coords, valid_paths):

            filename = os.path.basename(path)

            # optional truncation
            if len(filename) > 30:
                filename = filename[:27] + "..."

            ax.text(
                x0 + 5,
                y0 + h - 5,
                filename,
                color='white',
                fontsize=fontsize,
                ha='left',
                va='bottom',
                alpha=0.85,
                bbox=dict(
                    facecolor='black',
                    alpha=0.5,
                    pad=1,
                    edgecolor='none'
                )
            )

    # =========================
    # Save full resolution
    # =========================
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()

    print(f"Montage saved to: {output_path}")


# =========================
# Example usage
# =========================

if __name__ == "__main__":

    txt_path = r"C:\Users\Mikhalkino\Desktop\list_for_previews.txt"
    output_path = r"./montage_M0.png"
    parameters_path = r"./src/holodoppler/default_parameters.json"

    generate_m0_montage(txt_path, output_path, parameters_path)