from dataset import ImageDataset
from torch.utils.data import DataLoader
import torch
import matplotlib.pyplot as plt
import os
from solver import Solver

cwd = os.path.dirname(os.path.realpath(__file__))

# Can change to num_gpu * 4
num_workers = 0

train_images_dataset = ImageDataset(os.path.join(cwd, "data"), "train", paired=False)
test_images_dataset = ImageDataset(os.path.join(cwd, "data"), "test", paired=False)

train_images_dataloader = DataLoader(train_images_dataset, 16, True, num_workers=num_workers)
test_images_dataloader = DataLoader(test_images_dataset, 16, True, num_workers=num_workers)

device = "cuda" if torch.cuda.is_available() else "cpu"

# Setting up our model
criterion_GAN = torch.nn.MSELoss()
criterion_cycle = torch.nn.L1Loss()
criterion_identity = torch.nn.L1Loss()

model = Solver(
    criterion_cycle=criterion_cycle,
    criterion_GAN=criterion_GAN,
    criterion_identity=criterion_identity,
    input_shape= train_images_dataset[0]["M"].size(),
    device = device
)

model.train(train_images_dataloader, 1)

# Test print image
# dic = train_images_dataset[0]
# monet_test = dic['M'].permute(1, 2, 0)
# nature_test = dic['N'].permute(1, 2, 0)
# f, axarr = plt.subplots(2, 1)
# axarr[0].imshow(monet_test)
# axarr[1].imshow(nature_test)
# plt.show()