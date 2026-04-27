from holodoppler.Holodoppler import Holodoppler
import json
import os
import numpy as np
import matplotlib.pyplot as plt

with open(r"./src/holodoppler/default_parameters.json") as f :
    x = f.read()
    parameters = json.loads(x)
    
print("Using parameters :", parameters)


listpath = r"C:\Users\Ivashka\Desktop\list_for_zernike.txt"

for line in open(listpath, 'r'):
    HD = Holodoppler(backend = "cupyRAM", pipeline_version = "latest") # temp because of memory leak 

    print("Processing file :", line.strip())
    file_path = line.strip()
    HD.load_file(file_path)
    HD.process_moments_(parameters, holodoppler_path = True)