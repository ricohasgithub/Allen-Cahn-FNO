
import os
import io

import numpy as np
import imageio
import skimage.transform
import matplotlib.pyplot as plt

from tqdm import tqdm
from PIL import Image

import torch
from torch.utils.data import DataLoader, Dataset

'''
    Data loading classes
'''

class PairDataset(Dataset):

    def __init__(self, x, y):
        super(PairDataset, self).__init__()
        self.x = torch.Tensor(x)
        self.y = torch.Tensor(y)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        sample = {"X": self.x[idx], "Y": self.y[idx]}
        return sample

class DataModule():

    def __init__(self, data_dir, resize=(60, 60), max_data=None, n_test_simulation=12, batch_size=50,
                    skip_steps=10, store_steps_ahead=5, test_ratio=0.2):

        # Set class attributes
        self.data_dir = data_dir

        # Data constraints
        self.resize = resize
        self.max_data = max_data
        self.n_test_simulations = n_test_simulation
        self.batch_size = batch_size

        self.skip_steps = skip_steps
        self.store_steps_ahead = store_steps_ahead
        self.test_ratio = test_ratio

        # Load all data
        self.names, self.data_arrays = self.load_data()
        self.n_train = int((1 - self.test_ratio) * len(self.data_arrays))

        # Split training and test data
        self.arrays_train = self.data_arrays[:self.n_train]
        self.arrays_test = self.data_arrays[self.n_train:]

        self.x_train, self.y_train = self.prepare_x_y(self.arrays_train,
                                                        skip_steps = self.skip_steps,
                                                        store_steps_ahead = self.store_steps_ahead)

        self.train_dataset = PairDataset(self.x_train, self.y_train)


        self.x_test, self.y_test = self.prepare_x_y(self.arrays_test,
                                                        skip_steps = self.skip_steps,
                                                        store_steps_ahead = self.store_steps_ahead)

        self.test_dataset = PairDataset(self.x_test, self.y_test)
        self.test_simulations = self.select_test_simulations(self.arrays_test)

        self.train_dataloader = DataLoader(self.train_dataset, batch_size = self.batch_size, shuffle = True)
        self.val_dataloader = DataLoader(self.test_dataset, batch_size = self.batch_size, shuffle = True)


    def load_data(self):

        data_files = [name for name in os.listdir(self.data_dir) if "array" in name]

        if self.max_data:
            data_files = data_files[:self.max_data]

        data_arrays = [np.load(os.path.join(self.data_dir,file)) for file in tqdm(data_files)]

        if self.resize:
            for i,array in tqdm(enumerate(data_arrays)):
                data_arrays[i] = np.reshape(skimage.transform.resize(array,
                    (len(array), self.resize[0], self.resize[1])),
                    (len(array), 1, self.resize[0], self.resize[1]))

        return data_files, data_arrays

    def prepare_x_y(self, simulations, skip_steps = 10, store_steps_ahead = 5):

        X = []
        Y = []

        for simulation in tqdm(simulations):
            sim = simulation[2:] # Remove first spiky snapshots
            lsim = len(sim)

            for i in range(int(np.floor(lsim/skip_steps)) - store_steps_ahead):
                s = i * skip_steps
                _Y = []

                for j in range(1,store_steps_ahead):
                    sj = (i + j) * skip_steps
                    _Y.append(sim[sj])

                _Y = np.array(_Y)

                X.append(sim[s])
                Y.append(_Y)

        X = np.array(X)
        Y = np.array(Y)

        return X, Y

    def select_test_simulations(self, arrays_test):

        if len(arrays_test) < self.n_test_simulations:
            n_test_simulation = len(arrays_test)
        else:
            n_test_simulation = self.n_test_simulations

        test_simulations = [arrays_test[index][8:] for index in np.arange(0, n_test_simulation, 1)]

        extra_sims = [arrays_test[0],
                      arrays_test[1][5:],
                      arrays_test[2][15:]] #different u0s

        test_simulations.extend(extra_sims)

        return test_simulations

'''
    Model evaluation functions
'''

def eval_model_batch(model, test_batch):

    # Activate model eval time
    model.eval()

    with torch.no_grad():
        test_batch = torch.Tensor(test_batch)

        H, W = test_batch.shape[-2], test_batch.shape[-1]
        steps = test_batch.shape[1]
        batch = test_batch.shape[0]

        init = test_batch[0].view(batch, 1, H, W)

        x_eval = model(init)

        evals = []
        evals.append(np.array(x_eval.view(batch, H, W).cpu().detach().numpy()))

        for i in tqdm(range(1, steps)):
            x_eval = model(x_eval)
            evals.append(np.array(x_eval.view(batch, H, W).cpu().detach().numpy()))
        
        pred = np.array(evals)[:-1].transpose([1, 0, 2, 3])
        real = np.array(test_batch.cpu().detach().numpy()).reshape((batch, steps, H, W))[:, 1:, :]

        first = np.array(test_batch[:,0].view(batch, 1, H, W).cpu().detach().numpy())

        pred = np.concatenate((first, pred), axis = 1)
        real = np.concatenate((first, real), axis = 1)
        
    return pred, real

def eval_model(model, test_batch):

    # Activate model eval time
    model.eval()

    with torch.no_grad():
        test_batch = torch.Tensor(test_batch)

        H, W = test_batch.shape[-2], test_batch.shape[-1]
        steps = test_batch.shape[0]

        init = test_batch[0].view(1, 1, H, W)

        x_eval = model(init)

        evals = []
        evals.append(np.array(x_eval.view(H, W).cpu().detach().numpy()))

        for i in tqdm(range(1, len(test_batch))):
            x_eval = model(x_eval)
            evals.append(np.array(x_eval.view(H, W).cpu().detach().numpy()))
        
        pred = np.array(evals)[:-1]
        real = np.array(test_batch.cpu().detach().numpy()).reshape((len(test_batch), H, W))[1:]
        
        first = np.array(test_batch[0].view(1, H, W).cpu().detach().numpy())
        
        pred = np.concatenate((first, pred), axis = 0)
        real = np.concatenate((first, real), axis = 0)
        
    return pred, real

'''
    Animation and plotting
'''

def make_gif(list_ims, save_name, duration = 0.05, size = (200,200)):
    
    with imageio.get_writer(save_name,mode = "I", duration = duration) as writer:

        for sol in list_ims:
            s = sol
            im = ( (s-np.min(s))*(255.0/(np.max(s)-np.min(s))) ).astype(np.uint8)
            im = Image.fromarray(im).resize(size)
            writer.append_data(np.array(im))

    writer.close()

def fig_to_array(fig):
    
    io_buf = io.BytesIO()
    fig.savefig(io_buf, format='raw',quality = 95)
    io_buf.seek(0)
    img_arr = np.reshape(np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
                         newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1))
    io_buf.close()
    
    return img_arr

def make_simulation_gif(eval_sim, real_sim, name, duration = 1, skip_time = ""):
    
    assert np.shape(eval_sim) == np.shape(real_sim), "shapes_not equal"
    
    str_time = skip_time
    
    _max = np.max(real_sim)
    _min = np.min(real_sim)
    
    arrays = []
    for i in range(len(eval_sim)):
        
        if skip_time:
        
            str_time = "{} x dt".format(skip_time*i)
            
        im1 = eval_sim[i]
        im2 = real_sim[i]
        plt.close("all")
        fig = plt.figure()
        plt.subplot(121)
        plt.title("Predicted {}".format(str_time),fontdict = {"fontsize":22})
        o = plt.imshow(im1,cmap='gray', vmin=_min, vmax=_max)
        plt.axis('off')
        plt.subplot(122)
        plt.title("Real {}".format(str_time),fontdict = {"fontsize":22})
        o = plt.imshow(im2,cmap='gray', vmin=_min, vmax=_max)
        plt.axis('off')
        fig.tight_layout()
        
        array = fig_to_array(fig)
        arrays.append(array)
        
    make_gif(arrays, name , duration = duration)

def plot_phases(pred_sim, real_sim, results_dir, index = 0, epoch = "", name = None):
    
    try:
        os.makedirs(results_dir)
    except:
        pass
    
    fig = plt.figure()
    
    plt.subplot(121)
    means = [np.mean(np.abs(rs)) for rs in real_sim]
    plt.plot(means, label = "real")

    meansp = [np.mean(np.abs(ps)) for ps in pred_sim]
    plt.plot(meansp, label = "pred")

    plt.legend()

    plt.title("abs mean phase")

    plt.subplot(122)
    means = [np.mean(rs) for rs in real_sim]
    plt.plot(means, label = "real")

    meansp = [np.mean(ps) for ps in pred_sim]
    plt.plot(meansp, label = "pred")

    plt.legend()

    plt.title("abs  phase")
    
    if not(name):
        _name = "epoch {}".format(epoch)
        name = "{}_epoch_sim_{}_phases.png".format(epoch,index)
    else:
        _name = name
        name_dir = name+".PNG"
    fig.suptitle(_name)
    fig.savefig(os.path.join(results_dir,name_dir))
    
    return fig