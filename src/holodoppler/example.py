from holodoppler.Holodoppler import Holodoppler
from holodoppler.config import load_builtin_parameters
import os
import matplotlib.pyplot as plt

parameters = load_builtin_parameters().to_dict()
    
print("parameters :", parameters)

HD = Holodoppler(backend = "cupy", pipeline_version = "latest")

HD.load_file(r"D:\STAGE\260113_AUZ0752_2.holo")

print("file header :", HD.file_header)

# frames = HD.read_frames(0, 1)

# plt.imshow(frames[:,:,0])
# plt.show()

# HD.render_moments(parameters)

desktop = os.path.join(os.path.join(os.path.expanduser('~')), 'Desktop') 

HD.process_moments_(parameters, holodoppler_path = True)
