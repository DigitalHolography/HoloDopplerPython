from holodoppler.Holodoppler import Holodoppler
import json

from dotenv import load_dotenv
import os

load_dotenv()

with open(r"./parameters/default_parameters_lightest.json") as f :
    x = f.read()
    parameters = json.loads(x)
    
print("parameters :", parameters)

HD = Holodoppler(backend = "cupyRAM", pipeline_version = "latest")

HD.load_file(os.getenv("HOLOFILEDATA"))

print("file header :", HD.file_header)

HD.process_moments_(parameters, holodoppler_path = True)