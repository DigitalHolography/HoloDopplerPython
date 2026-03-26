from holodoppler.Holodoppler import Holodoppler
import json
import os
import matplotlib.pyplot as plt

with open("default_parameters.json") as f :
    x = f.read()
    parameters = json.loads(x)

parameters["spatial_propagation"] = "AngularSpectrum"
parameters["z"] = 0.4
parameters["pixel_pitch"] = 28e-6

parameters["end_frame"] = 2048

print("parameters :", parameters)

HD = Holodoppler(backend = "cupy")

HD.load_file("D:\\STAGE\\240214_MAO0581_OS1234_1.cine")

print("file metadata :", HD.cine_metadata_json)

# frames = HD.read_frames(0, 1).get()

# plt.imshow(frames[:,:,0])
# plt.show()

# HD.render_moments(parameters)

desktop = os.path.join(os.path.join(os.path.expanduser('~')), 'Desktop') 

HD.process_moments_(parameters, h5_path =  os.path.join(desktop,"temp.h5"), mp4_path = None, return_numpy = False)