import os
import re
import h5py
import numpy as np


def find_highest_hd_folder(parent_dir, name):
    pattern = re.compile(rf"{re.escape(name)}_HD_(\d+)$")
    max_num = -1
    best_folder = None

    for entry in os.listdir(parent_dir):
        full_path = os.path.join(parent_dir, entry)
        if not os.path.isdir(full_path):
            continue

        match = pattern.match(entry)
        if match:
            num = int(match.group(1))
            if num > max_num:
                max_num = num
                best_folder = full_path

    return best_folder


def process_holo_path(holo_path):
    parent_dir = os.path.dirname(holo_path)
    name = os.path.splitext(os.path.basename(holo_path))[0]

    hd_folder = find_highest_hd_folder(parent_dir, name)
    if not hd_folder:
        print(f"{name} : no HD folder found")
        return

    raw_path = os.path.join(hd_folder, "raw")
    if not os.path.isdir(raw_path):
        print(f"{name} : no raw folder")
        return

    # find first .h5 file
    h5_files = [f for f in os.listdir(raw_path) if f.endswith(".h5")]
    if not h5_files:
        print(f"{name} : no h5 file")
        return

    h5_path = os.path.join(raw_path, h5_files[0])

    with h5py.File(h5_path, "r") as f:
        if "zernike_coefs_radians" not in f:
            print(f"{name} : dataset missing")
            return

        data = f["zernike_coefs_radians"][:]  # shape: (time, modes)

    # average over time axis
    mean_vals = np.mean(data, axis=0)

    defocus = mean_vals[0] if len(mean_vals) > 0 else np.nan
    astig1  = mean_vals[1] if len(mean_vals) > 1 else np.nan
    astig2  = mean_vals[2] if len(mean_vals) > 2 else np.nan

    print(f"{name} : zernike coefs : defocus = {defocus:.6f} "
          f"astig1 = {astig1:.6f} astig2 = {astig2:.6f}")


def main(txt_file):
    with open(txt_file, "r") as f:
        lines = f.readlines()

    for line in lines:
        holo_path = line.strip()
        if not holo_path:
            continue

        process_holo_path(holo_path)


if __name__ == "__main__":
    txt_file = r"C:\Users\Ivashka\Desktop\list_for_zernike.txt"  # your txt file
    main(txt_file)