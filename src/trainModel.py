import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from src.model import EncoderDecoder
from src.solver import Solver
from torch.utils.data.sampler import SequentialSampler, SubsetRandomSampler
import pickle

config = {

    'data_path':        '', # Path to the parent directory of the image folder

    'continue_training':   False,      # Specify whether to continue training with an existing model and solver
    'start_epoch':            100,             # Specify the number of training epochs of the existing model
    'model_path': '',
    'solver_path': '',

    'do_overfitting': True,            # Set overfit or regular training

    'num_train_regular':    100000,     # Number of training samples for regular training
    'num_val_regular':      1000,       # Number of validation samples for regular training
    'num_train_overfit':    100,       # Number of training samples for overfitting test runs

    'num_workers': 4,                   # Number of workers for data loading

    'mode': 'pixel_interpolation',      # Define the image restoration task
    'min_max_corruption': [0, 1],      # Define the range of possible corruptions

    ## Hyperparameters ##
    'num_epochs': 1000,                  # Number of epochs to train
    'batch_size': 100,
    'learning_rate': 1e-3,
    'betas': (0.9, 0.999),              # Beta coefficients for ADAM
    'lr_decay': 1,                      # Learning rate decay -> lr *= lr_decay
    'lr_decay_interval': 1,             # Number of epochs after which to reduce the learning rate
    'num_subtasks': 5,                  # Number of subtasks for on-demand learning

    ## Logging ##
    'log_interval': 1,           # Number of mini-batches after which to print training loss
    'save_interval': 50,         # Number of epochs after which to save model and solver
    'save_path': '../saves'
}


data_set     = datasets.ImageFolder(config['data_path'], transform=transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor(), ]))

if config['do_overfitting']:
    train_data_sampler  = SequentialSampler(range(config['num_train_overfit']))
    val_data_sampler    = SequentialSampler(range(config['num_train_overfit']))

else:
    if config['num_train_regular']+config['num_val_regular'] > len(data_set):
        raise Exception('Trying to use more samples for training and validation than are available.')
    else:
        train_data_sampler  = SubsetRandomSampler(range(config['num_train_regular']))
        val_data_sampler    = SubsetRandomSampler(range(config['num_train_regular'], config['num_train_regular']+config['num_val_regular']))



train_data_loader   = torch.utils.data.DataLoader(dataset=data_set, batch_size=config['batch_size'], num_workers=config['num_workers'], sampler=train_data_sampler)
val_data_loader     = torch.utils.data.DataLoader(dataset=data_set, batch_size=config['batch_size'], num_workers=config['num_workers'], sampler=val_data_sampler)


if config['continue_training']:
    model = torch.load(config['model_path'])
    solver = pickle.load(open(config['solver_path'], 'rb'))
else:
    model = EncoderDecoder()
    solver = Solver(optim_args={"lr": config['learning_rate'],
                                "betas": config['betas']})

solver.train(lr_decay=config['lr_decay'],
             start_epoch=config['start_epoch'],
             model=model,
             train_loader=train_data_loader,
             val_loader=train_data_loader,
             num_epochs=config['num_epochs'],
             log_after_iters=config['log_interval'],
             save_after_epochs=config['save_interval'],
             lr_decay_interval=config['lr_decay_interval'],
             save_path=config['save_path'],
             num_subtasks=config['num_subtasks'])