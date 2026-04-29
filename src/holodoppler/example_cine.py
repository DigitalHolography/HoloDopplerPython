from holodoppler.Holodoppler import Holodoppler
import json
import os
import matplotlib.pyplot as plt

with open("default_parameters.json") as f :
    x = f.read()
    parameters = json.loads(x)

print("parameters :", parameters)

HD = Holodoppler(backend = "cupy")

HD.load_file("D:\\STAGE\\240214_MAO0581_OS1234_1.cine")

print("file metadata :", HD.cine_metadata_json)

HD.process_moments_(parameters, holodoppler_path=True)