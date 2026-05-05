from holodoppler.Holodoppler import Holodoppler
import json

from dotenv import load_dotenv
import os

load_dotenv()

with open(r"./parameters/default_parameters.json") as f :
    x = f.read()
    parameters = json.loads(x)
    
print("Using parameters :", parameters)

HD = Holodoppler(backend = "cupyRAM", pipeline_version = "latest")

listpath = os.getenv("TXTLISTFILE")

for line in open(listpath, 'r'):
    print("Processing file :", line.strip())
    file_path = line.strip()
    HD.load_file(file_path)
    HD.process_moments_(parameters, holodoppler_path = True)