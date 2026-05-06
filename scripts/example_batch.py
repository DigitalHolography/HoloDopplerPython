from holodoppler.Holodoppler import Holodoppler
import json

with open(r"./parameters/default_parameters.json") as f :
    x = f.read()
    parameters = json.loads(x)
    
print("Using parameters :", parameters)

HD = Holodoppler(backend = "cupyRAM", pipeline_version = "latest")

listpath = json.loads(open(r".debug_paths.json").read())["LISTTXTPATH"]

for line in open(listpath, 'r'):
    print("Processing file :", line.strip())
    file_path = line.strip()
    HD.load_file(file_path)
    HD.process_moments_(parameters, holodoppler_path = True)