from holodoppler.Holodoppler import Holodoppler
import json
import os
import matplotlib.pyplot as plt


with open(r"./src/holodoppler/default_parameters.json") as f :
    x = f.read()
    parameters = json.loads(x)
    
print("parameters :", parameters)

HD = Holodoppler(backend = "cupy", pipeline_version = "latest")

HD.load_file(r"D:\DATA_JULES\260423_EYE_TEST_9.holo")

print("file header :", HD.file_header)

frames = HD.read_frames(0, 1)
res = HD.render_moments(parameters, tictoc= False)

# main image
fig, ax = plt.subplots(1, 1, figsize=(20, 10))
ax.imshow(res["M0"].get())
ax.set_aspect('equal', adjustable='box')
plt.show()

HD.init_plot_debug()
dbg = HD.plot_debug(res, 0)

for key, img in dbg.items():
    dbg[key] = img.get() if hasattr(img, "get") else img

print("Debug images keys :", dbg.keys())
for k, img in dbg.items():
    plt.figure()
    plt.title(k)
    plt.imshow(img if img.ndim == 3 else img, cmap=None if img.ndim == 3 else "gray")
    plt.axis("off")
    plt.savefig(f"{k}.png", bbox_inches="tight")
    plt.close()