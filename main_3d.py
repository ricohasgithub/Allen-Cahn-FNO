
import os
import numpy as np

from re import S
from tabnanny import check
from tqdm import tqdm

import torch

from utils import DataModule, eval_model, eval_model_batch, make_simulation_gif, plot_phases
from fourier_neural_operator import Fourier_Net3D

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

def train(input_config, output_config, model_config):

    # Training constants
    scheduler_step = 100
    scheduler_gamma = 0.5

    ic, oc, mc = input_config, output_config, model_config
    results_root_dir, logs_dir, name_experiment = oc["results_root_dir"], oc["logs_dir"], oc["name_experiment"]

    if name_experiment == None:
        name_experiment = "experiment_" + mc["name_model"]

    # Prepare training data
    data_module = DataModule(ic["data_dir"], max_data = ic["max_data"])

    # Build model and configure training devices
    model = Fourier_Net3D(mc["modes_fourier"], mc["modes_fourier"], mc["modes_fourier"], mc["width_fourier"])

    optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)
    l1_loss = torch.nn.L1Loss()

    for epoch in range(mc["max_epochs"]):

        epoch_training_loss = []
        epoch_validation_loss = []

        # Training
        for i, batch in enumerate(data_module.train_dataloader):

            Xb, Ydata = batch["X"], batch["Y"][:,0:data_module.store_steps_ahead,:,:,:]
            
            Ypred = model(Xb).view(data_module.batch_size,
                                        data_module.skip_steps, data_module.skip_steps,
                                        data_module.store_steps_ahead)
            loss = l1_loss(Ypred, Ydata)
            Ypred = Ypred1

            losses = []
            losses.append(loss)

            Ypred = model(Ypred).view(data_module.batch_size,
                                        data_module.skip_steps, data_module.skip_steps,
                                        data_module.store_steps_ahead)
            losses.append(l1_loss(Ypred, Ydata[i]))

            losses = [l.view(1) for l in losses]
            loss = torch.mean(torch.cat(losses, 0))

            epoch_training_loss.append(loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Update learning rate with scheduler and output epoch training loss
        scheduler.step()
        mean_training_loss = torch.mean(torch.Tensor(epoch_training_loss))

        print("epoch: ", epoch, "training loss: ", mean_training_loss)

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

                epoch_validation_loss.append([val_loss1, val_loss2, val_loss3, val_loss4])

        validation_step_outputs = np.array(torch.mean(torch.Tensor(epoch_validation_loss), axis =0))
        print("epoch: ", epoch, "validation loss: ", validation_step_outputs[0])

    torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            }, "./models/model.pt")

def load_model(input_config, output_config, model_config):

    ic, oc, mc = input_config, output_config, model_config
    scheduler_step = 100
    scheduler_gamma = 0.5

    # Load Fourier_Net3D, optimizer, and scheduler
    model = Fourier_Net3D(mc["modes_fourier"], mc["modes_fourier"], mc["modes_fourier"], mc["width_fourier"])
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)

    # Load checkpoint file
    checkpoint = torch.load("./models/model.pt")

    # Load state dictionaries
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])

    return model, optimizer, scheduler

def eval_model_figs(model, data_module, results_dir):

    # Create new directory from results path (if it doesn't exist)
    try:
        os.makedirs(results_dir)
    except:
        pass

    # Fetch attributes from DataModule instance
    skip = data_module.skip_steps

    for i, test_sim in enumerate(data_module.test_simulations):

        # Create output file name (for gif)
        name = "sim_{}.gif".format(i)
        name = os.path.join(results_dir, name)

        test_sim = test_sim[::skip]
        pred_sim, real_sim = eval_model(model, test_sim)
        
        # Generate figures
        make_simulation_gif(pred_sim, real_sim, name, duration = 1, skip_time = skip)
        plot_phases(pred_sim, real_sim, results_dir, index = i)

if __name__ == "__main__":

    # DataModule for evaluating test simulations
    data_module = DataModule(input_config["data_dir"], max_data = input_config["max_data"])

    train(input_config, output_config, model_config)
    model, optimizer, scheduler = load_model(input_config, output_config, model_config)
    eval_model_figs(model, data_module, "./results")