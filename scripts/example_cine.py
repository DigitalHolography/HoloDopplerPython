from holodoppler.Holodoppler import Holodoppler
import json

with open(r"./parameters/default_parameters_cine.json") as f :
    x = f.read()
    parameters = json.loads(x)

print("parameters :", parameters)

HD = Holodoppler(backend = "cupyRAM")

HD.load_file(json.loads(open(r".debug_paths.json").read())["CINEFILEPATH"])

print("file metadata :", HD.cine_metadata_json)

HD.process_moments_(parameters, holodoppler_path=True)