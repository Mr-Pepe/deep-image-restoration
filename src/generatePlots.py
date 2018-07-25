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
from src.solver import Solver

def show_images(images):
    to_pil = transforms.ToPILImage()

    for image in images:
        plt.imshow(to_pil(image))
        plt.axis('off')
        plt.show()

def show_images_in_grid(images):
    to_pil = transforms.ToPILImage()

    for i, image in enumerate(images):
        plt.subplot(images.shape[0],1,i+1)
        plt.imshow(to_pil(image))
        plt.axis('off')

    plt.show()



celeba_path = '/home/felipe/Projects/CelebA_dataset/'
sun397_path = '/home/felipe/Projects/SUN397/'
db11_path = '/home/felipe/Projects/DB11'

db11_set            = datasets.ImageFolder(db11_path, transform=transforms.Compose([transforms.Resize((128,128)), transforms.ToTensor(), ]))
sun397_set          = datasets.ImageFolder(sun397_path, transform=transforms.Compose([transforms.Resize((128,128)), transforms.ToTensor(), ]))
celeba_trainset     = datasets.ImageFolder(celeba_path, transform=transforms.Compose([transforms.Resize((128,128)), transforms.ToTensor(), ]))
train_data_loader   = torch.utils.data.DataLoader(dataset=celeba_trainset, batch_size=100, num_workers=4)
val_data_loader     = torch.utils.data.DataLoader(dataset=celeba_trainset, batch_size=100, num_workers=4)


model_path = "../saves/train20180708185328/model1080"
model = torch.load(model_path)

solver_path = '../saves/train20180708185328/solver1080'


corrupter = Corrupter([0, 15, 30, 45, 60, 75], 'pixel_interpolation')

# Images from train set
batch = next(iter(train_data_loader))
batch = batch[0]

ground_truth = batch[0:3]
corrupted = corrupter(ground_truth.clone(), np.array([1,0,1,0,1]))
out = model(corrupted)

# show_images_in_grid(ground_truth)
# show_images_in_grid(corrupted)
# show_images_in_grid(out)

criterion = torch.nn.MSELoss(reduce=False)
loss = criterion(out, ground_truth)
loss = loss.cpu().detach().numpy()
den = loss.shape[1]*loss.shape[2]*loss.shape[3]
loss = loss.sum(3).sum(2).sum(1)/den

print(loss)

psnr = 10*np.log10(1/loss)

print(psnr)

# Images from val set
# batch = next(iter(val_data_loader))
# batch = batch[0]
#
# show_images(corrupter(batch[10:50], np.array([5,5,5,5,5])))

# PSNR
# error = np.arange(1e-5, 0.02, 1e-5)
# psnr = 10*np.log10(1/error)
# plt.plot(error, psnr)
# plt.xlabel('MSE')
# plt.ylabel('PSNR')
# plt.axes().xaxis.label.set_fontsize(20)
# plt.axes().yaxis.label.set_fontsize(20)
# [x.set_fontsize(12) for x in plt.axes().get_xticklabels()]
# [x.set_fontsize(12) for x in plt.axes().get_yticklabels()]
# plt.show()


# Solver

solver = pickle.load(open(solver_path, 'rb'))

loss = np.array(solver.train_loss_history)
psnr = np.array(solver.psnr_history)
num_per_subtask = np.array(solver.num_per_subtask_history)
start = 91

batchsize = num_per_subtask[start:].sum(axis=1)

plt.plot(num_per_subtask[start:,0]/batchsize)
plt.plot(num_per_subtask[start:,1]/batchsize)
plt.plot(num_per_subtask[start:,2]/batchsize)
plt.plot(num_per_subtask[start:,3]/batchsize)
plt.plot(num_per_subtask[start:,4]/batchsize)
plt.xlabel("Epochs")
plt.ylabel("Samples per subtask")
plt.show()

