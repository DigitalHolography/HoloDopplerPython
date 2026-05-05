from holodoppler.Holodoppler import Holodoppler
import json
import os
import matplotlib.pyplot as plt
import h5py
import numpy as np
import matplotlib.pyplot as plt

with open("C:\\Users\\Ivashka\\Documents\\Python\\pyflow\\src\\holodoppler\\default_parameters.json") as f :
    x = f.read()
    parameters = json.loads(x)
    
print("parameters :", parameters)

HD = Holodoppler(backend = "cupy", pipeline_version="old")

holo_path = "D:\\STAGE\\260113_AUZ0752_2.holo"

HD.load_file(holo_path)
print("file header :", HD.file_header)

desktop = os.path.join(os.path.join(os.path.expanduser('~')), 'Desktop') 

test_path = os.path.join(desktop,"tempold.h5") # from python compute

HD.process_moments_(parameters, h5_path =  test_path, mp4_path = None, return_numpy = False)

ref_path = "D:\\STAGE\\260113_AUZ0752_2_HD_2\\raw\\260113_AUZ0752_2_HD_2_output.h5" # from matlab compute

with h5py.File(test_path, "r") as f_test, h5py.File(ref_path, "r") as f_ref:
    test_data = f_test["moment0"][:,:,:]
    ref_data = np.permute_dims(np.flip(f_ref["moment0"][:,0,:,:], axis=1),axes=(2,1,0))
    # test_reg = np.array(f_test["registration"][:])
    # ref_reg = np.array(f_ref["registration"][:])
plt.figure()
plt.imshow(np.mean(test_data,axis=2))
plt.colorbar()
plt.figure()
plt.imshow(np.mean(ref_data,axis=2))
plt.colorbar()


# Check shape
if test_data.shape != ref_data.shape:
    raise ValueError(f"Shape mismatch: {test_data.shape} vs {ref_data.shape}")

# Absolute difference
abs_diff = np.abs(test_data - ref_data)

plt.figure()
plt.imshow(abs_diff[:,:,0])
plt.colorbar()

# Percentage difference (relative to reference)
# Avoid division by zero
with np.errstate(divide='ignore', invalid='ignore'):
    percent_diff = np.where(
        ref_data != 0,
        abs_diff / np.abs(ref_data) * 100,
        0.0
    )


print("=== Absolute Difference ===")
print(f"Min:  {abs_diff.min()}")
print(f"Max:  {abs_diff.max()}")
print(f"Mean: {abs_diff.mean()}")

print("\n=== Percentage Difference (%) ===")
print(f"Min:  {percent_diff.min()}")
print(f"Max:  {percent_diff.max()}")
print(f"Mean: {percent_diff.mean()}")

plt.show()