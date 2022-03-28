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

data_module = DataModule(input_config["data_dir"], max_data = input_config["max_data"])
reader = torch.from_numpy(data_module.x_test)
reader = torch.reshape(reader, (300, S, S, 1))

# load data
test_a = reader[:,::sub,::sub, 3:T_in*4:4]
test_u = reader[:,::sub,::sub, indent+T_in*4:indent+(T+T_in)*4:sub_t]

print(test_a.shape, test_u.shape)

# pad the location information (s,t)
S = S * (4//sub)
T = T * (4//sub_t)

test_a = test_a.reshape(ntest,S,S,1).repeat([1,1,1,T])

gridx = torch.tensor(np.linspace(0, 1, S), dtype=torch.float)
gridx = gridx.reshape(1, S, 1, 1, 1).repeat([1, 1, S, T, 1])
gridy = torch.tensor(np.linspace(0, 1, S), dtype=torch.float)
gridy = gridy.reshape(1, 1, S, 1, 1).repeat([1, S, 1, T, 1])
gridt = torch.tensor(np.linspace(0, 1, T+1)[1:], dtype=torch.float)
gridt = gridt.reshape(1, 1, 1, T, 1).repeat([1, S, S, 1, 1])

test_a = torch.cat((gridx.repeat([ntest,1,1,1,1]), gridy.repeat([ntest,1,1,1,1]),
                       gridt.repeat([ntest,1,1,1,1]), test_a), dim=-1)

t2 = default_timer()
print('preprocessing finished, time used:', t2-t1)
device = torch.device('cuda')

# load model
model = torch.load('models/model.pt')
l1_loss = torch.nn.L1Loss()

print(model.count_params())

# test
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u), batch_size=1, shuffle=False)
myloss = torch.nn.L1Loss()
pred = torch.zeros(test_u.shape)
index = 0

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
