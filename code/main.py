from dataset import ImageDataset
from torch.utils.data import DataLoader
import torch
import matplotlib.pyplot as plt
import os
import numpy as np
from solver import Solver
import gc
import random

cwd = os.path.dirname(os.path.realpath(__file__))

gc.collect()
torch.cuda.empty_cache()

# TRAINING PARAMETERS
load = None
# load = "weights-13-04-2023-23-00"
epochs = 100
num_workers = 0 # Can change to num_gpu * 4
batch_size = 1


train_images_dataset = ImageDataset(os.path.join(cwd, "data"), "train", paired=False)
test_images_dataset = ImageDataset(os.path.join(cwd, "data"), "test", paired=False)

train_images_dataloader = DataLoader(train_images_dataset, batch_size, True, num_workers=num_workers)
test_images_dataloader = DataLoader(test_images_dataset, batch_size, True, num_workers=num_workers)

device = "cuda" if torch.cuda.is_available() else "cpu"

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

model.train(train_images_dataloader, test_images_dataloader, epochs=epochs, save_model=True)


# Test print image
f, axarr = plt.subplots(2, 8, constrained_layout=True)
f.set_figheight(8)
f.set_figwidth(16)

for i in range(8):
    eval_image = test_images_dataset[random.randint(0, len(test_images_dataset) - 1)]["N"]
    generated_image = model.eval(eval_image.unsqueeze(0)).squeeze()
    generated_image = (generated_image * 255).to(torch.uint8)

    axarr[0][i].imshow(eval_image.permute(1, 2, 0))
    axarr[0][i].axis("off")
    axarr[0][i].set_title("Input Image:")
    axarr[1][i].imshow(generated_image.permute(1, 2, 0))
    axarr[1][i].axis("off")
    axarr[1][i].set_title("Image w/ Monet style:")

plt.show()