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


num_train_overfit = 1000
overfit_sampler = SequentialSampler(range(num_train_overfit))

num_train_regular = 100000
train_data_sampler  = SubsetRandomSampler(range(num_train_regular))
val_data_sampler    = SubsetRandomSampler(range(num_train_regular,101000))

# celeba_path = '/home/felipepeter/CelebA_dataset/'
celeba_path = '/home/felipe/Projects/CelebA_dataset/'

# cifar10_trainset    = datasets.CIFAR10(root='../datasets', train=True, download=True, transform=transforms.ToTensor())
celeba_trainset     = datasets.ImageFolder(celeba_path, transform=transforms.Compose([transforms.Resize((64,64)), transforms.ToTensor(), ]))



train_data_loader   = torch.utils.data.DataLoader(dataset=celeba_trainset, batch_size=300, num_workers=4, sampler=overfit_sampler)
val_data_loader     = torch.utils.data.DataLoader(dataset=celeba_trainset, batch_size=10, num_workers=4, sampler=overfit_sampler)

model = EncoderDecoder()
solver = Solver(optim_args={"lr":0.001,
                            "betas": (0.9, 0.999)})

# model = torch.load('../saves/train20180708111322/model570')
# solver = pickle.load(open('../saves/train20180708111322/solver570', 'rb'))
# solver.lr = 1e-3

solver.train(lr_decay=1, start_epoch=0, model=model, train_loader=train_data_loader, val_loader=train_data_loader, num_epochs=1500, log_after_iters=1, save_after_epochs=10)