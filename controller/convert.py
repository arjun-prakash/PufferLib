
import torch
import pandas as pd


# load either a Module or a raw state_dict
model_path = '/users/apraka15/arjun/puffer/PufferLib/experiments/puffer_drone_crazyflie_175468010708/model_puffer_drone_crazyflie_000096.pt'

import torch
from gaussian_mlp import generate

# Load your model or state_dict

model_or_sd = torch.load(model_path, map_location="cpu")

# Dump out C code
c_code = generate(model_or_sd, output_path="network_evaluate.c")
print("Wrote network_evaluate.c")
