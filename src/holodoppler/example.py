from holodoppler.Holodoppler import Holodoppler
import json
import os
import matplotlib.pyplot as plt

with open("C:\\Users\\Ivashka\\Documents\\Python\\HolodopplerPython\\src\\holodoppler\\default_parameters.json") as f :
    x = f.read()
    parameters = json.loads(x)
    
print("parameters :", parameters)

HD = Holodoppler(backend = "cupy", pipeline_version = "old")

HD.load_file("D:\\STAGE\\260113_AUZ0752_2.holo")

print("file header :", HD.file_header)

# frames = HD.read_frames(0, 1)

# plt.imshow(frames[:,:,0])
# plt.show()

# HD.render_moments(parameters)

desktop = os.path.join(os.path.join(os.path.expanduser('~')), 'Desktop') 

HD.process_moments_(parameters, holodoppler_path = True)