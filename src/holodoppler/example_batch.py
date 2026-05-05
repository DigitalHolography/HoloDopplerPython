from holodoppler.Holodoppler import Holodoppler
import json
import os
import numpy as np
import matplotlib.pyplot as plt

with open(r"./src/holodoppler/default_parameters_debug.json") as f :
    x = f.read()
    parameters = json.loads(x)
    
print("Using parameters :", parameters)

HD = Holodoppler(backend = "cupy", pipeline_version = "latest")


listpath = r"C:\Users\Ivashka\Desktop\list_for_zernike.txt"

for line in open(listpath, 'r'):
    print("Processing file :", line.strip())
    file_path = line.strip()
    HD.load_file(file_path)
    HD.process_moments_(parameters, holodoppler_path = True)