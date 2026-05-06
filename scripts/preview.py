from holodoppler.Holodoppler import Holodoppler
import json
import os
import matplotlib.pyplot as plt
import numpy as np
from matlab_imresize.imresize import imresize

parameter_path = "./parameters/default_parameters_debug.json"
holo_path = json.loads(open(r".debug_paths.json").read())["HOLOFILEPATH"]

with open(parameter_path) as f :
    x = f.read()
    parameters = json.loads(x)
    
print("parameters :", parameters)

HD = Holodoppler(backend = "cupyRAM", pipeline_version = "latest")

HD.load_file(holo_path)

print("file header :", HD.file_header)

frames = HD.read_frames(0, 1)
res = HD.render_moments(parameters, tictoc= False)

def plot_debug_safe(HD, res):
    HD.init_plot_debug(parameters)
    debug = {}

    for key, plotter in HD.debug_plotters.items():
        try:
            args = HD.debug_sources[key](res)
        except KeyError:
            continue

        debug[key] = plotter.plot(*args)

    return debug

import imageio.v3 as iio

def save_debug_images(debug_dict, save_dir, prefix="debug"):
    os.makedirs(save_dir, exist_ok=True)

    for key, img in debug_dict.items():
        if img is None:
            continue

        img_np = HD._to_numpy(img)

        if img_np.ndim == 2 and parameters["square"]:
            H, W = img_np.shape
            L = max(H, W)
            # --- Resize ---
            img_np = imresize(img_np, output_shape=(L, L))

        if img_np.dtype != np.uint8:
            img_min = np.min(img_np)
            img_max = np.max(img_np)

            if img_max > img_min:
                img_np = (img_np - img_min) / (img_max - img_min + 1e-12)

            img_np = (img_np * 255).astype(np.uint8)

        filename = os.path.join(save_dir, f"{prefix}_{key}.png")

        iio.imwrite(filename, img_np)

        print(f"Saved: {filename} | shape={img_np.shape} dtype={img_np.dtype}")


# --- Generate debug safely ---
debug_imgs = plot_debug_safe(HD, res)

if parameters["debug"] and parameters["shack_hartmann"] and parameters["shack_hartmann_zernike_fit"]:
    print("zernike_fit_coeffs (radians):", HD._to_numpy(res["coefs"]) if "coefs" in res else "N/A")
    # Assuming coef[0] is defocus, calculate the corresponding delta z in mm
    defocus_coef = HD._to_numpy(res["coefs"])[0] if "coefs" in res else 0
    R = min(frames.shape[1:]) * parameters["pixel_pitch"] / 2
    
    # Nx = frames.shape[2]
    # Ny = frames.shape[1]
    # dx_out = parameters["wavelength"] * parameters["z"] / (Nx * parameters["pixel_pitch"])
    # dy_out = parameters["wavelength"] * parameters["z"] / (Ny * parameters["pixel_pitch"])
    # D_phys = min(Nx * dx_out, Ny * dy_out) /2
    
    # print(D, D_phys)
    K = (
        2.0 * np.sqrt(3.0)
        * parameters["wavelength"]
        / (np.pi * R**2)
    )

    z_corr = 1.0 / (1.0 / parameters["z"] - K * defocus_coef)

    delta_z_mm = (z_corr - parameters["z"]) * 1e3
    print("delta to true z in mm if coef[0] is defocus:", delta_z_mm)

# --- Add M0 ---
if "M0" in res:
    M0 = HD._to_numpy(res["M0"])
    M0 = (M0 - np.min(M0)) / (np.max(M0) - np.min(M0) + 1e-12)
    debug_imgs["M0"] = (M0 * 255).astype(np.uint8)

print("DEBUG KEYS:", list(debug_imgs.keys()))

# --- Save ---
save_dir = "./debug_outputs"
save_debug_images(debug_imgs, save_dir)