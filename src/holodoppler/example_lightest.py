from holodoppler.Holodoppler import Holodoppler
import json
import os
import matplotlib.pyplot as plt

with open(r"./src/holodoppler/default_parameters_lightest.json") as f :
    x = f.read()
    parameters = json.loads(x)
    
print("parameters :", parameters)

HD = Holodoppler(backend = "cupy", pipeline_version = "latest")

HD.load_file(r"D:\PROJETS\DATA\260113_AUZ0752_6.holo") #"D:\za\260429_zzz.holo")

print("file header :", HD.file_header)

# frames = HD.read_frames(0, 1)

# plt.imshow(frames[:,:,0])
# plt.show()

# HD.render_moments(parameters)

desktop = os.path.join(os.path.join(os.path.expanduser('~')), 'Desktop') 

HD.process_moments_(parameters, holodoppler_path = True)