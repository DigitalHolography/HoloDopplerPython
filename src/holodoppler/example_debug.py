from holodoppler.Holodoppler import Holodoppler
import json
import os
import matplotlib.pyplot as plt

with open(r"./src/holodoppler/default_parameters_debug.json") as f :
    x = f.read()
    parameters = json.loads(x)
    
print("parameters :", parameters)

HD = Holodoppler(backend = "cupy", pipeline_version = "latest")

HD.load_file(r"D:\za\260429_zzz.holo")

print("file header :", HD.file_header)

HD.process_moments_(parameters, holodoppler_path = True)