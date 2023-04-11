from dataset import ImageDataset
from torch.utils.data import DataLoader
import torch
import matplotlib.pyplot as plt
import os
import numpy as np
from solver import Solver

import random

cwd = os.path.dirname(os.path.realpath(__file__))


import gc
gc.collect()
torch.cuda.empty_cache()

# Can change to num_gpu * 4
num_workers = 0

train_images_dataset = ImageDataset(os.path.join(cwd, "data"), "train", paired=False)
test_images_dataset = ImageDataset(os.path.join(cwd, "data"), "test", paired=False)

train_images_dataloader = DataLoader(train_images_dataset, 16, True, num_workers=num_workers)
test_images_dataloader = DataLoader(test_images_dataset, 16, True, num_workers=num_workers)

device = "cuda" if torch.cuda.is_available() else "cpu"


# TRAINING PARAMETERS
load = None
load = "weights-11-04-2023-00-31"
epochs = 10

print(device)

# Setting up our model
criterion_GAN = torch.nn.MSELoss()
criterion_cycle = torch.nn.L1Loss()
criterion_identity = torch.nn.L1Loss()

model = Solver(
    criterion_cycle=criterion_cycle,
    criterion_GAN=criterion_GAN,
    criterion_identity=criterion_identity,
    input_shape= train_images_dataset[0]["M"].size(),
    device = device,
    load= load
)

# model.train(train_images_dataloader, test_images_dataloader, epochs=epochs, save_model=True)
eval_image = test_images_dataset[random.randint(0, len(test_images_dataset) - 1)]["N"]
generated_image = model.eval(eval_image.unsqueeze(0)).squeeze()

# print(eval_image)
# print(generated_image)

# Test print image
generated_image = (generated_image * 255).to(torch.uint8)
plot_image = generated_image.permute(1, 2, 0)
f, axarr = plt.subplots(2, 1)
axarr[0].imshow(plot_image)
axarr[1].imshow(eval_image.permute(1, 2, 0))
plt.show()