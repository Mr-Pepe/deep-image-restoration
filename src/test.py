import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from PIL import Image
import torch
import torchvision
import torchvision.datasets as datasets

from src.model import EncoderDecoder
from src.solver import Solver
from src.utils import delete_pixels


cifar10_trainset = datasets.CIFAR10(root='../datasets', train=True, download=True, transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
train_data_loader = torch.utils.data.DataLoader(dataset=cifar10_trainset, batch_size=100, num_workers=4)

# image_mean = torch.stack([t.mean(1).mean(1) for t, c in cifar10_trainset]).mean(0)

# model = EncoderDecoder()

# model_path = "/home/felipe/Dropbox/Master/Semester_02/Seminar/project/saves/train20180627131912/model1"
# model = torch.load(model_path)

batch = next(iter(train_data_loader))[0]

batch = delete_pixels(batch, 0.8)

to_pil = torchvision.transforms.ToPILImage()

plt.imshow(to_pil(batch[0]))


solver = Solver(optim_args={"lr":1e-3})
#
# solver.train(model=model, train_loader=train_data_loader, num_epochs=10, \
#              log_after_iters=100, save_after_epochs=1)