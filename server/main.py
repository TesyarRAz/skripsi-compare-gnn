import torch

from typing import Union

from fastapi import FastAPI

import model
import usecase
import utils
from ml_model import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.serialization.safe_globals([torch.nn.DataParallel])

model_filepaths = [(f"./assets/model/{x}/tsp_{type}.pt", ResidualGatedGCNModel if type == 'gcn' else ResidualGATModel if type == 'gat' else ResidualGATv2Model) for x in ['10_30', '10_50', '20_30', '20_50'] for type in ['gcn', 'gat', 'gat_v2']]

models = {}
for filepath, model_class in model_filepaths:
    net = torch.nn.DataParallel(model_class(variables,  torch.float32, torch.long)).to(device)
    net.load_state_dict(torch.load(filepath, map_location=device))
    models['/'.join(filepath.split('/')[-2:][::-1]).replace('.pt', '').replace('tsp_', '')] = net

print(f"Models loaded: {list(models.keys())}")

app = FastAPI()

@app.post("/predict/coord")
def predict_with_coord(tour: model.PredictCoordTour):
    """
    Endpoint untuk memprediksi rute TSP dengan koordinat node.
    """
    return usecase.predict_with_coords(models, device, tour)