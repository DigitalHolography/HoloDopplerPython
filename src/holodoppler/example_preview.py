from holodoppler.Holodoppler import Holodoppler
import json
import os
import matplotlib.pyplot as plt


with open(r"./src/holodoppler/default_parameters.json") as f :
    x = f.read()
    parameters = json.loads(x)
    
print("parameters :", parameters)

HD = Holodoppler(backend = "cupy", pipeline_version = "latest")

HD.load_file(r"D:\STAGE\260113_AUZ0752_2.holo")

print("file header :", HD.file_header)

frames = HD.read_frames(0, 1)
res = HD.render_moments(parameters, tictoc= True)

fig, ax = plt.subplots(1,1, figsize=(20,10))

ax.imshow(res["M0"].get())
plt.show()
# desktop = os.path.join(os.path.join(os.path.expanduser('~')), 'Desktop') 

# HD.process_moments_(parameters, holodoppler_path = True)