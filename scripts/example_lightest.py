from holodoppler.Holodoppler import Holodoppler
import json

with open(r"./parameters/default_parameters_lightest.json") as f :
    x = f.read()
    parameters = json.loads(x)
    
print("parameters :", parameters)

HD = Holodoppler(backend = "cupyRAM", pipeline_version = "latest")

HD.load_file(json.loads(open(r".debug_paths.json").read())["HOLOFILEPATH"])

print("file header :", HD.file_header)

HD.process_moments_(parameters, holodoppler_path = True)