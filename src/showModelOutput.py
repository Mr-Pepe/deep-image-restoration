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

from src.utils import Corrupter


def evalModel(images):

    for image in images:
        to_pil = transforms.ToPILImage()

        plt.subplot(1,3,1)
        plt.imshow(to_pil(image))

        restored = model(torch.unsqueeze(image, 0))

        plt.subplot(1,3,2)
        plt.imshow(to_pil(restored[0]))

        plt.subplot(1, 3, 3)
        plt.imshow(to_pil(restored[0]-image))
        plt.show()


plt.interactive(False)

celeba_path = '/home/felipe/Projects/CelebA_dataset/'

num_train_overfit = 100
overfit_sampler = SequentialSampler(range(num_train_overfit))
num_train_regular = 100000
train_data_sampler  = SubsetRandomSampler(range(num_train_regular))
val_data_sampler    = SubsetRandomSampler(range(num_train_regular,101000))


# cifar10_trainset    = datasets.CIFAR10(root='../datasets', train=True, download=True, transform=transforms.ToTensor())
# train_data_loader   = torch.utils.data.DataLoader(dataset=cifar10_trainset, batch_size=100, num_workers=4, sampler=overfit_sampler)
#
# cifar10_testset = datasets.CIFAR10(root='../datasets', train=False, download=True, transform=transforms.ToTensor())
# test_data_loader = torch.utils.data.DataLoader(dataset=cifar10_testset, batch_size=100, num_workers=4)

celeba_trainset     = datasets.ImageFolder(celeba_path, transform=transforms.Compose([transforms.Resize((64,64)), transforms.ToTensor(), ]))
train_data_loader = torch.utils.data.DataLoader(dataset=celeba_trainset, batch_size=100, num_workers=4, sampler=overfit_sampler)
val_data_loader     = torch.utils.data.DataLoader(dataset=celeba_trainset, batch_size=100, num_workers=4, sampler=val_data_sampler)


model_path = "/home/felipe/Projects/seminar/saves/train20180709141228/model230"
model = torch.load(model_path)
# model = EncoderDecoder()

corrupter = Corrupter([0, 15, 30, 45, 60, 75], 'pixel_interpolation')

batch = next(iter(val_data_loader))
batch = batch[0]

batch = corrupter(batch[30:60], np.array([5,5,5,5,5]))

evalModel(batch[0:30])




