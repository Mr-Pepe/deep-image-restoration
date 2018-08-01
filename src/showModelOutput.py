import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from PIL import Image, ImageChops
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data.sampler import SequentialSampler, SubsetRandomSampler
from src.model import EncoderDecoder
import pickle
from src.utils import eval_model

from src.utils import Corrupter

config = {

    'data_path': '', # Path to the parent directory of the image folder

    'model_path': '',

    'do_overfitting': True,            # Set overfit or regular training

    'val_set_indices': range(100),      # Indices of the images to load for testing
    'num_show_images': 10,                 # Number of images to show

    'num_workers': 4,                   # Number of workers for data loading

    'mode': 'pixel_interpolation',      # Define the image restoration task
    'min_max_corruption': [0, 1],      # Define the range of possible corruptions

    'num_subtasks': 5,                  # Number of subtasks
}



data_set = datasets.ImageFolder(config['data_path'], transform=transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor(), ]))

data_sampler  = SequentialSampler(config['val_set_indices'])

data_loader   = torch.utils.data.DataLoader(dataset=data_set, batch_size=100, num_workers=config['num_workers'], sampler=data_sampler)

model = torch.load(config['model_path'])

sampling_points = np.linspace(config['min_max_corruption'][0], config['min_max_corruption'][1], config['num_subtasks']+1)
corrupter = Corrupter(config['mode'], sampling_points)

batch = next(iter(data_loader))
batch = batch[0]

batch = corrupter(batch, np.array(np.ones((config['num_subtasks'],)) * data_loader.batch_size / config['num_subtasks']))

eval_model(model, batch[np.linspace(0,99,config['num_show_images'], dtype=int).tolist()])




