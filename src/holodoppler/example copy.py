from holodoppler.Holodoppler import Holodoppler
import json
import os
import matplotlib.pyplot as plt

with open("C:\\Users\\Ivashka\\Documents\\Python\\HolodopplerPython\\src\\holodoppler\\default_parameters.json") as f :
    x = f.read()
    parameters = json.loads(x)
    
print("Using parameters :", parameters)

HD = Holodoppler(backend = "cupy", pipeline_version = "old")


listpath = "path/to/list.txt"

for line in open(listpath, 'r'):
    file_path = line.strip()
    HD.load_file(file_path)
    HD.process_moments_(parameters, holodoppler_path = True)