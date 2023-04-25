from model import Generator, Discriminator
import numpy as np
import torch
import itertools
from typing import Dict, List, Tuple
from tqdm.auto import tqdm
from torch.autograd import Variable

import datetime
import os

import matplotlib.pyplot as plt
    

def format_file_name(dir, dataset_name):
    today = datetime.datetime.now()
    filename = today.strftime("%d-%m-%Y-%H-%M")

    filename = f"weights-{filename}"
    cwd = os.path.dirname(os.path.realpath(__file__))

    parent_directory_path = os.path.join(cwd, "weights", dataset_name)
    
    if not os.path.isdir(parent_directory_path):
        print(parent_directory_path)
        os.makedirs(parent_directory_path)
    
    child_directory_path = os.path.join(parent_directory_path, dir)

    if not os.path.isdir(child_directory_path):
        print(child_directory_path)
        os.makedirs(child_directory_path)

    weight_path = os.path.join(cwd, "weights", dataset_name, dir, filename)
    return weight_path

class Solver():
    def __init__(self, criterion_GAN, criterion_cycle, criterion_identity, device, input_shape, dataset_name,load=None, from_dataset_name="nature"):
        self.device = device
        self.input_shape = input_shape

        # Hyper parameters
        self.lr = 0.0002
        self.batch_size = 4
        self.b1 = 0.5
        self.b2 = 0.999
        self.decay_epoch = 1
        self.n_workers = 8
        self.lambda_cyc = 10.0
        self.lambda_id = 5.0
        self.dataset_name = dataset_name
        self.from_dataset_name = from_dataset_name

        self.criterion_GAN = criterion_GAN.to(device)
        self.criterion_cycle = criterion_cycle.to(device)
        self.criterion_identity = criterion_identity.to(device)
        self.GeneratorMN = Generator(input_shape=input_shape, num_residual_blocks=3).to(device)
        self.GeneratorNM = Generator(input_shape=input_shape, num_residual_blocks=3).to(device)

        self.DiscriminatorM = Discriminator(input_shape=input_shape).to(device)
        self.DiscriminatorN = Discriminator(input_shape=input_shape).to(device)

        if load:
            self.load_weights(self.GeneratorMN, f"generator_{self.from_dataset_name}", load)
            self.load_weights(self.GeneratorNM, f"generator_{dataset_name}", load)
            self.load_weights(self.DiscriminatorN, f"discriminator_{from_dataset_name}", load)
            self.load_weights(self.DiscriminatorM, f"discriminator_{dataset_name}", load)

        
        self.optimizer_G = torch.optim.Adam(
            itertools.chain(self.GeneratorMN.parameters(), self.GeneratorNM.parameters()),
            lr = self.lr, betas=(self.b1, self.b2)
            )
        
        self.optimizer_D_M = torch.optim.Adam(self.DiscriminatorM.parameters(), lr=self.lr, betas=(self.b1, self.b2))
        self.optimizer_D_N = torch.optim.Adam(self.DiscriminatorN.parameters(), lr=self.lr, betas=(self.b1, self.b2))

    def train_step(self, dataloader, epoch, results):
        # Setup test loss and test accuracy values
        test_loss, test_acc = 0, 0
        loss_gen = loss_id = loss_gan = loss_cyc = 0.0
        loss_disc = loss_disc_m = loss_disc_n = 0.0

        tqdm_bar = tqdm(dataloader, desc=f"Training Epoch {epoch} ", total=int(len(dataloader)), mininterval=(int(len(dataloader))/1000))

        
        for batch_idx, batch  in enumerate(tqdm_bar):
            # Send data to target device
            monet_real, nature_real = batch["M"].to(self.device), batch["N"].to(self.device)

            true_labels = Variable(
                torch.Tensor(np.ones((monet_real.size(0), *self.DiscriminatorM.output_shape))),
                requires_grad=False,
            )
            false_labels = Variable(
                torch.Tensor(np.zeros((monet_real.size(0), *self.DiscriminatorM.output_shape))),
                requires_grad=False,
            )

            true_labels = true_labels.to(self.device)
            false_labels = false_labels.to(self.device)

            #TRAINING GENERATOR
            self.GeneratorMN.train()
            self.GeneratorNM.train()
            self.optimizer_G.zero_grad()

            # Identity Loss
            g_loss_id_M = self.criterion_identity(self.GeneratorNM(monet_real), monet_real)
            g_loss_id_N = self.criterion_identity(self.GeneratorMN(nature_real), nature_real)
            g_loss_identity = (g_loss_id_M + g_loss_id_N) / 2
            
            # Gan loss
            '''
            We train our Gan loss by creating false images and putting it through the discrimator of the 
            translated domain. We then put it through the loss function with the ground truth being true_labels
            as the goal of our gans network is to fool the discriminator
            '''
            monet_fake = self.GeneratorNM(nature_real)

            # if batch_idx == (int(len(dataloader)) - 1):
            #     plt.figure()
            #     plt.imshow(monet_fake[0].detach().cpu().squeeze().permute((1,2,0)))
            #     plt.show()
            
            g_loss_gan_M = self.criterion_GAN(self.DiscriminatorM(monet_fake), true_labels)

            nature_fake = self.GeneratorMN(monet_real)
            g_loss_gan_N = self.criterion_GAN(self.DiscriminatorN(nature_fake), true_labels)
            
            g_loss_gan = (g_loss_gan_M + g_loss_gan_N) / 2
            
            # Cycle loss
            reconstructed_M = self.GeneratorNM(nature_fake)
            g_loss_cyc_M = self.criterion_cycle(reconstructed_M, monet_real)

            reconstructed_N = self.GeneratorMN(monet_fake)
            g_loss_cyc_N = self.criterion_cycle(reconstructed_N, nature_real)

            g_loss_cyc = (g_loss_cyc_M + g_loss_cyc_N) / 2
            
            # Calculate total loss and backprop to update gradient
            loss_G = self.lambda_id * g_loss_identity + g_loss_gan + self.lambda_cyc * g_loss_cyc
            loss_G.backward(retain_graph=True)
            self.optimizer_G.step()
            
            # TRAINING DISCRIMINATOR
            
            # Train Monet discriminator
            self.DiscriminatorM.train()
            self.optimizer_D_M.zero_grad()
            
            # Real loss
            d_real_loss_m = self.criterion_GAN(self.DiscriminatorM(monet_real), true_labels)
            # Fake loss
            d_fake_loss_m = self.criterion_GAN(self.DiscriminatorM(monet_fake), false_labels)

            loss_d_m =  (d_real_loss_m + d_fake_loss_m) / 2
            loss_d_m.backward()
            self.optimizer_D_M.step()

            # Train Nature discriminator
            self.DiscriminatorN.train()
            self.optimizer_D_N.zero_grad()

            # Real loss
            d_real_loss_n = self.criterion_GAN(self.DiscriminatorN(nature_real), true_labels)
            # Fake loss
            d_fake_loss_n = self.criterion_GAN(self.DiscriminatorN(nature_fake), false_labels)

            loss_d_n =  (d_real_loss_n + d_fake_loss_n) / 2
            loss_d_n.backward()
            self.optimizer_D_N.step()
        
            loss_d_total = (loss_d_m + loss_d_n) / 2

            #Prints
            loss_gen += loss_G.item()
            loss_id += g_loss_identity.item()
            loss_cyc += g_loss_cyc.item()

            loss_disc += loss_d_total.item()
            loss_disc_m += loss_d_m.item()
            loss_disc_n += loss_d_n.item()
            
            results["g_total_loss"].append(loss_gen / (batch_idx + 1))
            results["g_identity_loss"].append(loss_id / (batch_idx + 1))
            results["cycle_loss"].append(loss_cyc / (batch_idx + 1))
            results["d_total_loss"].append(loss_disc / (batch_idx + 1))
            results["d_m_loss"].append(loss_disc_m / (batch_idx + 1))
            results["d_n_loss"].append(loss_disc_n / (batch_idx + 1))
            
            tqdm_bar.set_postfix(Gen_loss=loss_gen / (batch_idx + 1),
                                 identity = loss_id / (batch_idx + 1),
                                 adv = loss_gan / (batch_idx + 1), 
                                 cycle = loss_cyc / (batch_idx + 1),
                                 disc_loss = loss_disc / (batch_idx + 1),
                                 d_m_loss = loss_disc_m / (batch_idx + 1),
                                 d_n_loss = loss_disc_n / (batch_idx + 1)
                                 )
        
    def test_step(self, dataloader, epoch, results):
        # Setup test loss and test accuracy values
        loss_gen = loss_id = loss_gan = loss_cyc = 0.0
        loss_disc = loss_disc_m = loss_disc_n = 0.0

        tqdm_bar = tqdm(dataloader, desc=f"Testing Epoch {epoch} ", total=int(len(dataloader)), mininterval=(int(len(dataloader))/1000))

        
        for batch_idx, batch  in enumerate(tqdm_bar):
            # Send data to target device
            monet_real, nature_real = batch["M"].to(self.device), batch["N"].to(self.device)
            
            true_labels = torch.Tensor(np.ones((monet_real.size(0), 1)))
            true_labels = Variable(true_labels, requires_grad=False).to(self.device)

            false_labels = torch.Tensor(np.zeros((monet_real.size(0), 1)))
            false_labels = Variable(false_labels, requires_grad=False).to(self.device)
            
            self.GeneratorMN.eval()
            self.GeneratorNM.eval()
            self.DiscriminatorM.eval()
            self.DiscriminatorN.eval()

            #Testing GENERATOR

            # Identity Loss
            g_loss_id_M = self.criterion_identity(self.GeneratorNM(monet_real), monet_real)
            g_loss_id_N = self.criterion_identity(self.GeneratorMN(nature_real), nature_real)
            g_loss_identity = (g_loss_id_M + g_loss_id_N) / 2
            
            # Gan loss
            '''
            We train our Gan loss by creating false images and putting it through the discrimator of the 
            translated domain. We then put it through the loss function with the ground truth being true_labels
            as the goal of our gans network is to fool the discriminator
            '''
            monet_fake = self.GeneratorNM(nature_real)
            
            g_loss_gan_M = self.criterion_GAN(self.DiscriminatorM(monet_fake), true_labels)

            nature_fake = self.GeneratorMN(monet_real)
            g_loss_gan_N = self.criterion_GAN(self.DiscriminatorN(nature_fake), true_labels)
            
            g_loss_gan = (g_loss_gan_M + g_loss_gan_N) / 2
            
            # Cycle loss
            reconstructed_M = self.GeneratorNM(nature_fake)
            g_loss_cyc_M = self.criterion_cycle(reconstructed_M, monet_real)

            reconstructed_N = self.GeneratorMN(monet_fake)
            g_loss_cyc_N = self.criterion_cycle(reconstructed_N, nature_real)

            g_loss_cyc = (g_loss_cyc_M + g_loss_cyc_N) / 2
            
            # Calculate total loss and backprop to update gradient
            loss_G = self.lambda_id * g_loss_identity + g_loss_gan + self.lambda_cyc * g_loss_cyc
            
            # TESTING DISCRIMINATOR
            
            # Testing Monet discriminator
            
            d_real_loss_m = self.criterion_GAN(self.DiscriminatorM(monet_real), true_labels) # Real loss
            d_fake_loss_m = self.criterion_GAN(self.DiscriminatorM(monet_fake), false_labels) # Fake loss

            loss_d_m = (d_real_loss_m + d_fake_loss_m) / 2

            # Train Nature discriminator

            d_real_loss_n = self.criterion_GAN(self.DiscriminatorN(nature_real), true_labels) # Real loss
            d_fake_loss_n = self.criterion_GAN(self.DiscriminatorN(nature_fake), false_labels) # Fake loss

            loss_d_n =  (d_real_loss_n + d_fake_loss_n) / 2
            loss_d_total = (loss_d_m + loss_d_n) / 2

            #Prints
            loss_gen += loss_G.item()
            loss_id += g_loss_identity.item()
            loss_cyc += g_loss_cyc.item()

            loss_disc += loss_d_total.item()
            loss_disc_m += loss_d_m.item()
            loss_disc_n += loss_d_n.item()

            results["g_total_loss"].append(loss_gen / (batch_idx + 1))
            results["g_identity_loss"].append(loss_id / (batch_idx + 1))
            results["cycle_loss"].append(loss_cyc / (batch_idx + 1))
            results["d_total_loss"].append(loss_disc / (batch_idx + 1))
            results["d_m_loss"].append(loss_disc_m / (batch_idx + 1))
            results["d_n_loss"].append(loss_disc_n / (batch_idx + 1))

            tqdm_bar.set_postfix(Gen_loss=loss_gen / (batch_idx + 1),
                                 identity = loss_id / (batch_idx + 1),
                                 adv = loss_gan / (batch_idx + 1), 
                                 cycle = loss_cyc / (batch_idx + 1),
                                 disc_loss = loss_disc / (batch_idx + 1),
                                 d_m_loss = loss_disc_m / (batch_idx + 1),
                                 d_n_loss = loss_disc_n / (batch_idx + 1)
                                 )
   
    def train(self,
          train_dataloader: torch.utils.data.DataLoader, 
          test_dataloader: torch.utils.data.DataLoader,
          epochs: int,
          save_model = False
          ):

        # Create empty results dictionary
        train_results = {
                "g_total_loss": [],
                "g_identity_loss": [],
                "cycle_loss": [],
                "d_total_loss": [],
                "d_m_loss" : [],
                "d_n_loss" : [],
        }
        
        test_results = {
                "g_total_loss": [],
                "g_identity_loss": [],
                "cycle_loss": [],
                "d_total_loss": [],
                "d_m_loss" : [],
                "d_n_loss" : [],
        }

        # Loop through training and testing steps for a number of epochs
        for epoch in tqdm(range(epochs)):
            self.train_step( dataloader=train_dataloader, epoch=epoch+1, results=train_results)
            self.test_step(dataloader=test_dataloader, epoch=epoch+1, results=test_results)

            if save_model:
                self.save_discriminator(self.dataset_name)
                self.save_discriminator(self.from_dataset_name)
                self.save_generator(self.dataset_name)
                self.save_generator(self.from_dataset_name)
                
                print("Successfully saved")
        
        return train_results, test_results
    
    def eval(self, input_image):
        self.GeneratorNM.eval()
        input_image = input_image.to(self.device)
        generated_image = self.GeneratorNM(input_image)
        
        return generated_image.detach().to("cpu")
    
    def save_discriminator(self, model_to_save="monet"):
        model = self.DiscriminatorM
        dir = f"discriminator_{model_to_save}"
        if model_to_save == "nature":
            model = self.DiscriminatorN
        
        save_path = format_file_name(dir, self.dataset_name)
        torch.save(model.state_dict(), save_path)

    
    def save_generator(self, model_to_save="monet"):
        model = self.GeneratorNM
        dir = f"generator_{model_to_save}"
        if model_to_save == "nature":
            model = self.GeneratorMN
        
        save_path = format_file_name(dir, self.dataset_name)
        torch.save(model.state_dict(), save_path)
    
    # This requires you to have a weights folder
    def load_weights(self, model, dir, filename):
        cwd = os.path.dirname(os.path.realpath(__file__))
        weight_path = os.path.join(cwd, "weights", self.dataset_name, dir, filename)
        model.load_state_dict(torch.load(weight_path))
        