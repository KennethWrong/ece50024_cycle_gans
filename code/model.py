import torch.nn as nn
import torch.nn.functional as F
import torch

def conv(c_in, c_out, k_size, stride=1, pad=0, bn=True):
    layers = []
    layers.append(nn.Conv2d(c_in, c_out, k_size, stride, pad, bias=True))
    if bn:
        layers.append(nn.BatchNorm2d(c_out))
    return layers


# Residual block w/ 2 3x3 conv layers with same # of filters on both layers
class ResidualBlock(nn.Module):
    def __init__(self, in_channel):
        super(ResidualBlock, self).__init__()

        self.res_block = nn.Sequential(
            nn.ReflectionPad2d(1),
            *conv(c_in=in_channel, c_out=in_channel, k_size=3),
            # nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=3),
            nn.InstanceNorm2d(num_features= in_channel),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            *conv(c_in=in_channel, c_out=in_channel, k_size=3),
            # nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=3),
            nn.InstanceNorm2d(num_features= in_channel)
        )
    
    def forward(self, input):
        return self.res_block(input) + input

# 7x7 Convolution, InstanceNorm,, ReLU layer w k filters and stride 1
class C7S1_K(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(C7S1_K, self).__init__()

        self.block = nn.Sequential(
            nn.ReflectionPad2d(in_channel),
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=7),
            nn.InstanceNorm2d(num_features= out_channel),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, input):
        return self.block(input)

# 3x3 Convolution, InstanceNorm, ReLU layer w/ k filters and stride 2
class D_K(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(D_K, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(num_features= out_channel),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, input):
        return self.block(input)

class U_K(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(U_K, self).__init__()

        self.block = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(num_features= out_channel),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, input):
        return self.block(input)


class Generator(nn.Module):
    """Generator for transfering from mnist to svhn"""
    def __init__(self, input_shape, num_residual_blocks = 9):
        super(Generator, self).__init__()
        # encoding blocks
        input_channel , input_width, input_height = input_shape

        model = [
            C7S1_K(in_channel=3, out_channel=64),
            D_K(in_channel=64, out_channel= 128),
            D_K(in_channel=128, out_channel= 256)
        ]
        
        for _ in range(num_residual_blocks):
            model += [ResidualBlock(in_channel= 256)]
        
        
        model += [
           U_K(in_channel=256, out_channel=128),
           U_K(in_channel=128, out_channel=64),
        ]
        
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=64, out_channels=3, kernel_size=7),
            nn.Tanh()
        ]
        
        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)


def discriminator_block(c_in, c_out, normal=False):
    layers = []
    layers += [nn.Conv2d(c_in, c_out, kernel_size=4, stride=2, padding=1)]
    
    if normal:
        layers += [nn.InstanceNorm2d(c_out)] 
    layers += [nn.LeakyReLU(0.2, inplace=True)]
    return layers

'''
The idea behind this Discriminator is that instead of the traditional GANS generator that outputs a binary classifaction
of real or fake. PatchGan outputs a matrix of 0s and 1s to determine which patch is real and which patch is fake.

Then based on this we compare it to the true/false labels (matrix of 0s and 1s of same size) to calculate loss.
'''
class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()

        channels, height, width = input_shape

        # Output dimensions of PatchGAN
        self.output_shape = (1, height // 2 ** 4, width // 2 ** 4)

        # C64 -> C128 -> C256 -> C512
        self.model = nn.Sequential(
            *discriminator_block(channels, c_out=64),
            *discriminator_block(64, c_out=128, normal=True),
            *discriminator_block(128, c_out=256, normal=True),
            *discriminator_block(256, c_out=512, normal= True),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, padding=1)
        )

    def forward(self, img):
        return self.model(img)