from holodoppler.Holodoppler import Holodoppler
import json

from dotenv import load_dotenv
import os

load_dotenv()

with open(r"./parameters/default_parameters_cine.json") as f :
    x = f.read()
    parameters = json.loads(x)

print("parameters :", parameters)

HD = Holodoppler(backend = "cupyRAM")

HD.load_file(os.getenv("CINEFILEDATA"))

print("file metadata :", HD.cine_metadata_json)

HD.process_moments_(parameters, holodoppler_path=True)