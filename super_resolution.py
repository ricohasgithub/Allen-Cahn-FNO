import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt

from utils import DataModule, eval_model, eval_model_batch, make_simulation_gif, plot_phases
from fourier_neural_operator import Fourier_Net2D

import operator
from functools import reduce
from functools import partial

from timeit import default_timer
import scipy.io

torch.manual_seed(0)
np.random.seed(0)

t1 = default_timer()

TEST_PATH = 'data/ns_data_V1e-4_N20_T50_test.mat'


ntest = 20

sub = 1
sub_t = 1
S = 60
T_in = 10
T = 20

indent = 1

# Prepare training data

# Model params
output_config = {
    "results_root_dir": "./results",
    "save_every": 10,
    "logs_dir": "logs",
    "name_experiment": None,
}

input_config = {
    "skip_steps": 6,
    "max_data": None,
    "data_dir": "./data/AC",
    "gpus": -1,
}

model_config = {
    "max_epochs": 100,
    "early_stopping_patience": 10,
    "early_stopping_min_delta": 1e-4,
    "lr": 5*1e-4,
    "normalization": False,
    "n_blocks": 6,
    "layers_per_block": 3,
    "channels": 70,
    "name_model": "fourier",
    "skip_con_weight": 0.1,
    "modes_fourier": 16,
    "width_fourier": 120,
}

# Spatial super resolution resize tensor
sub = 1
S = 60 * (4//sub)
resize_tensor = (S, S)

data_module = DataModule(input_config["data_dir"], max_data = input_config["max_data"],
                         resize=resize_tensor)

# Load model
model = Fourier_Net2D(model_config["modes_fourier"], model_config["modes_fourier"], model_config["width_fourier"])
# Load checkpoint file
checkpoint = torch.load("./models/model.pt")
# Load state dictionaries
model.load_state_dict(checkpoint["model"])

l1_loss = torch.nn.L1Loss()

validation_loss = []

# Validation
with torch.no_grad():

    for i, batch in enumerate(data_module.val_dataloader):

        Xb, Ystep1, Ystep2, Ystep3, Ystep4 = batch["X"], batch["Y"][:,0,:,:,:], batch["Y"][:,1,:,:,:], batch["Y"][:,2,:,:,:], batch["Y"][:,2,:,:,:]
        Ypred1 = model(Xb)
        Ypred2 = model(Ypred1)
        Ypred3 = model(Ypred2)
        Ypred4 = model(Ypred3)

        val_loss1 = l1_loss(Ypred1, Ystep1)
        val_loss2 = l1_loss(Ypred2, Ystep2)
        val_loss3 = l1_loss(Ypred3, Ystep3)
        val_loss4 = l1_loss(Ypred4, Ystep4)

        validation_loss.append([val_loss1, val_loss2, val_loss3, val_loss4])
