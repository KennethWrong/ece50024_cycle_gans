import numpy as np
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


def plot_losses(results, title):
    # Plot training error
    train_g_total_loss = results["g_total_loss"] 
    train_g_identity_loss = results["g_identity_loss"] 
    train_cycle_loss = results["cycle_loss"] 
    train_d_total_loss = results["d_total_loss"] 
    train_d_m_loss = results["d_m_loss"] 
    train_d_n_loss = results["d_n_loss"] 

    x_axis = np.arange(1, len(train_g_total_loss) + 1, 1)

    f, axarr = plt.subplots(2, 3, constrained_layout=True)
    f.suptitle(title)
    f.set_figheight(8)
    f.set_figwidth(16)

    labels = ["g_total_loss", "g_identity_loss", "cycle_loss", "discr_total_loss",
            "discr_monet_loss", "discr_nature_loss"]
    
    losses = [train_g_total_loss, train_g_identity_loss, train_cycle_loss, train_d_total_loss,
            train_d_m_loss, train_d_n_loss]

    for i in range(2):
        for j in range(3):
            index = i*3 + j
            axarr[i][j].plot(x_axis, losses[index])
            axarr[i][j].set_title(labels[index])
            axarr[i][j].set_xlabel("Iterations")
            axarr[i][j].set_ylabel("Loss")
            
    plt.show()

def plot_testing_images(eval_dataset_dataloader, model, from_im, to_im = ""):
    
    f, axarr = plt.subplots(2, 8, constrained_layout=True)
    f.set_figheight(8)
    f.set_figwidth(16)
    for i, batch in enumerate(eval_dataset_dataloader):
        eval_image = batch[from_im]
        generated_image = model.eval(eval_image).squeeze().permute(1,2,0).cpu().detach().numpy()
        # generated_image = (generated_image * 255).astype(np.int32)

        axarr[0][i].imshow(eval_image.squeeze().permute(1, 2, 0))
        axarr[0][i].axis("off")
        axarr[0][i].set_title("Input Image:")
        axarr[1][i].imshow(generated_image)
        axarr[1][i].axis("off")
        axarr[1][i].set_title(f"{to_im} style")

        if i == 7:
            break

    plt.show()

def train_image_dataset(image_set, from_image_set, load, epochs, num_workers, batch_size, save_model):
    train_images_dataset = ImageDataset(os.path.join(cwd, "data"), "train", image_set, from_image_set, paired=False)
    test_images_dataset = ImageDataset(os.path.join(cwd, "data"), "test", image_set, from_image_set, paired=False)

    train_images_dataloader = DataLoader(train_images_dataset, batch_size, True, num_workers=num_workers)
    test_images_dataloader = DataLoader(test_images_dataset, batch_size, False, num_workers=num_workers)

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
        dataset_name=image_set,
        from_dataset_name=from_image_set,
        device = device,
        load= load
    )

    train_results, test_results = model.train(train_images_dataloader, test_images_dataloader, epochs=epochs, save_model=save_model)
    # # Plot training and testing error
    # plot_losses(train_results, "Training error of Cycle GANs")
    # plot_losses(test_results, "Testing error of Cycle GANs")

    # Plotting evaluation images
    # eval_images_dataset = ImageDataset(os.path.join(cwd, "data"), "paper_test", image_set, from_image_set, paired=False)
    eval_images_dataset = ImageDataset(os.path.join(cwd, "data"), "test", image_set, from_image_set, paired=False)
    eval_images_dataloader = DataLoader(eval_images_dataset, batch_size, False, num_workers=num_workers)
    plot_testing_images(eval_images_dataloader, model, "N", image_set)

if __name__ == "__main__":
    from_image_sets = ["nature"] # Input image set
    to_image_sets = ["monet"] # Target style transformation
    loads = [None] # This is for loading weight files
    epochs = 30
    num_workers = 0 # Can change to num_gpu * 4
    batch_size = 1
    save_model = True # If you want to save the weights
    
    for i in range(len(to_image_sets)):
        gc.collect()
        torch.cuda.empty_cache()
        train_image_dataset(image_set= to_image_sets[i],
                            from_image_set= from_image_sets[i],
                            load = loads[i],
                            epochs= epochs,
                            num_workers= num_workers,
                            save_model= save_model,
                            batch_size= batch_size)