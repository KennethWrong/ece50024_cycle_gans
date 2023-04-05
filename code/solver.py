from model import Generator, Discriminator
import numpy as np
import torch
import itertools
from typing import Dict, List, Tuple
from tqdm.auto import tqdm
from torch.autograd import Variable

class Solver():
    def __init__(self, criterion_GAN, criterion_cycle, criterion_identity, device, input_shape):
        self.device = device
        self.input_shape = input_shape

        # Hyper parameters
        self.lr = 0.00012
        self.batch_size = 4
        self.b1 = 0.5
        self.b2 = 0.999
        self.decay_epoch = 1
        self.n_workers = 8
        self.lambda_cyc = 10.0
        self.lambda_id = 5.0

        self.criterion_GAN = criterion_GAN.to(device)
        self.criterion_cycle = criterion_cycle.to(device)
        self.criterion_identity = criterion_identity.to(device)

        self.GeneratorMN = Generator(input_shape=input_shape, num_residual_blocks=1).to(device)
        self.GeneratorNM = Generator(input_shape=input_shape, num_residual_blocks=1).to(device)

        self.DiscriminatorM = Discriminator(input_shape=input_shape).to(device)
        self.DiscriminatorN = Discriminator(input_shape=input_shape).to(device)

        
        self.optimizer_G = torch.optim.Adam(
            itertools.chain(self.GeneratorMN.parameters(), self.GeneratorNM.parameters()),
            lr = self.lr, betas=(self.b1, self.b2)
            )
        self.optimizer_D_M = torch.optim.Adam(self.DiscriminatorM.parameters(), lr=self.lr, betas=(self.b1, self.b2))
        self.optimizer_D_N = torch.optim.Adam(self.DiscriminatorN.parameters(), lr=self.lr, betas=(self.b1, self.b2))

    def train_step(self, dataloader, epoch):
        # Setup test loss and test accuracy values
        test_loss, test_acc = 0, 0
        loss_gen = loss_id = loss_gan = loss_cyc = 0.0
        loss_disc = loss_disc_m = loss_disc_n = 0.0

        tqdm_bar = tqdm(dataloader, desc=f"Training Epoch {epoch} ", total=int(len(dataloader)))

        
        for batch_idx, batch  in enumerate(dataloader):
            # Send data to target device
            monet_real, nature_real = batch["M"].to(self.device), batch["N"].to(self.device)
            
            true_labels = torch.Tensor(np.ones((monet_real.size(0), *self.DiscriminatorM.output_shape)))
            true_labels = Variable(true_labels, requires_grad=False)

            false_labels = torch.Tensor(np.zeros((monet_real.size(0), *self.DiscriminatorM.output_shape)))
            false_labels = Variable(false_labels, requires_grad=False)

            #TRAINING GENERATOR
            self.GeneratorMN.train()
            self.GeneratorNM.train()
            self.optimizer_G.zero_grad()

            # Identity Loss

            print("Calculating Identity Loss")
            g_loss_id_M = self.criterion_identity(self.GeneratorNM(monet_real), monet_real)
            g_loss_id_N = self.criterion_identity(self.GeneratorMN(nature_real), nature_real)
            g_loss_identity = (g_loss_id_M + g_loss_id_N) / 2
            
            # Gan loss
            '''
            We train our Gan loss by creating false images and putting it through the discrimator of the 
            translated domain. We then put it through the loss function with the ground truth being true_labels
            as the goal of our gans network is to fool the discriminator
            '''
            print("Calculating GAN Loss")
            monet_fake = self.GeneratorNM(nature_real)
            g_loss_gan_M = self.criterion_GAN(self.DiscriminatorM(monet_fake), true_labels)

            nature_fake = self.GeneratorMN(monet_real)
            g_loss_gan_N = self.criterion_GAN(self.DiscriminatorN(nature_fake), true_labels)

            g_loss_gan = (g_loss_gan_M + g_loss_gan_N) / 2
            
            # Cycle loss
            print("Calculating Cycle Loss")
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
            print("Calculating Monet Discriminator Loss")
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
            print("Calculating Nature Discriminator Loss")
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

            tqdm_bar.set_postfix(Gen_loss=loss_gen / (batch_idx + 1),
                                 identity = loss_id / (batch_idx + 1),
                                 adv = loss_gan / (batch_idx + 1), 
                                 cycle = loss_cyc / (batch_idx + 1),
                                 discriminator_loss = loss_disc / (batch_idx + 1),
                                 d_m_loss = loss_disc_m / (batch_idx + 1),
                                 d_n_loss = loss_disc_n / (batch_idx + 1)
                                 )
        
            # if (batch_idx % 50 == 0):
                # print(f"Epoch: {epoch} [{batch_idx*len(test_pred_labels)}/{len(dataloader)*len(test_pred_labels)}] total_train_acc: {test_acc / (batch_idx + 1)}")

        # Adjust metrics to get average loss and accuracy per batch 
        # test_loss = test_loss / len(dataloader)
        # test_acc = test_acc / len(dataloader)
        # return test_loss, test_acc
        
    
    
    def train(self,
          train_dataloader: torch.utils.data.DataLoader, 
        #   test_dataloader: torch.utils.data.DataLoader, 
          epochs: int,
          ):

        # Create empty results dictionary
        results = {"train_loss": [],
                "train_acc": [],
                "test_loss": [],
                "test_acc": []
        }
        
        # Loop through training and testing steps for a number of epochs
        for epoch in tqdm(range(epochs)):
            self.train_step( dataloader=train_dataloader, epoch=epoch+1)

            # Print out what's happening
            # print(
            #     f"Epoch: {epoch+1} | "
            #     f"train_loss: {train_loss:.4f} | "
            #     f"train_acc: {train_acc:.4f} | "
            #     f"test_loss: {test_loss:.4f} | "
            #     f"test_acc: {test_acc:.4f}"
            # )

            # # Update results dictionary
            # results["train_loss"].append(train_loss)
            # results["train_acc"].append(train_acc)
            # results["test_loss"].append(test_loss)
            # results["test_acc"].append(test_acc)

            # model_class.save_model()
            # print("Successfully saved")

        # Return the filled results at the end of the epochs
        return results
        


        
        