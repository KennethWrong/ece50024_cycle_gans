import torch.nn as nn
import torch.nn.functional as F
import torch


def deconv(c_in, c_out, k_size, stride=2, pad=1, output_pad=0, bn=True):
    """Custom deconvolutional layer for simplicity."""
    layers = []
    layers.append(nn.ConvTranspose2d(c_in, c_out, k_size, stride, pad, output_padding=output_pad, bias=False))
    if bn:
        layers.append(nn.BatchNorm2d(c_out))
    return nn.Sequential(*layers)

def conv(c_in, c_out, k_size, stride=2, pad=1, bn=True):
    """Custom convolutional layer for simplicity."""
    layers = []
    layers.append(nn.Conv2d(c_in, c_out, k_size, stride, pad, bias=False))
    if bn:
        layers.append(nn.BatchNorm2d(c_out))
    return nn.Sequential(*layers)

class Generator(nn.Module):
    """Generator for transfering from mnist to svhn"""
    def __init__(self, input_shape):
        super(Generator, self).__init__()
        # encoding blocks
        input_channel , input_width, input_height = input_shape
        
        # input shape: [N, 3, 224, 224]
        
        self.conv1 = conv(c_in=input_channel, c_out=16, k_size=2, stride=2, pad=8) # shape: [N, 64, 120, 120]
        self.conv2 = conv(c_in=16, c_out=32, k_size=4, stride=2, pad=1) # shape: [N, 32, 60, 60]

        # residual blocks
        self.conv3 = conv(c_in=32, c_out=16, k_size=3, stride=1, pad=1) # shape: [N, 16, 60, 60]
        self.conv4 = conv(c_in=16, c_out=1, k_size=4, stride=2, pad=1) # shape: [N, 1, 30, 30]

        # Flatten to [N, 900]

        # FCNN
        self.fc1 = nn.Linear(in_features=900, out_features=3600) # shape: [N, 3600]

        # Reshape to [N, 1, 60, 60]
        
        # decoding blocks
        self.upsample1 = nn.Upsample(scale_factor=2, mode="bilinear") # shape: [N, 1, 120, 120]
        self.reflection1 = nn.ReflectionPad2d(1) # shape: [N, 1, 122, 122]
        self.deconv1 = conv(c_in=1, c_out=2, k_size=3, stride=1, pad=1) # shape: [n, 2, 122, 122]

        self.upsample2 = nn.Upsample(scale_factor=1.5, mode="bilinear") #shape: [N, 2, 183, 183]
        self.reflection2 = nn.ReflectionPad2d(2) #shape: [N, 2, 185, 185]
        self.deconv2 = conv(c_in=2, c_out=3, k_size=3, stride=1, pad=1) # shape: [n, 3, 185, 185]

        self.upsample3 = nn.Upsample(scale_factor=1.2, mode="bilinear") # shape: [n, 3, 222, 222]
        self.reflection3 = nn.ReflectionPad2d(1) #shape: [N, 3, 224, 224]
        self.deconv3 = conv(c_in=3, c_out=3, k_size=3, stride=1, pad=0) # shape: [n, 3, 224, 224]

        

    def forward(self, x):
        # ENCODING
        out = F.leaky_relu(self.conv1(x), 0.1)      
        out = F.leaky_relu(self.conv2(out), 0.1)    
        out = F.leaky_relu(self.conv3(out), 0.1)    
        out = F.leaky_relu(self.conv4(out), 0.1)    

        out = torch.flatten(out, start_dim=1, end_dim=-1)

        out = F.leaky_relu(self.fc1(out), 0.1) 
        
        out = torch.reshape(out, (-1, 1, 60, 60)) 
        
        # DECODING
        out = self.upsample1(out)
        out = self.reflection1(out)
        out = F.leaky_relu(self.deconv1(out), 0.1)
        
        out = self.upsample2(out)
        out = self.reflection2(out)
        out = F.leaky_relu(self.deconv2(out), 0.1)

        out = self.upsample3(out)
        out = self.reflection3(out)
        
        out = torch.tanh(self.deconv3(out))

        # print(f"After deconv2: {out.shape}")

        return out

class Discriminator(nn.Module):
    """Discriminator for svhn."""
    def __init__(self, input_shape, conv_dim=64):
        super(Discriminator, self).__init__()
        conv_dim = 64

        # input shape: [N, 3, 224, 224]

        self.conv1 = conv(c_in= 3, c_out= conv_dim, k_size= 4, stride= 2, pad= 1 ,bn= False) #[N, 64, 112, 112]
        self.conv2 = conv(c_in= conv_dim, c_out= 2*conv_dim, k_size= 4, stride= 2, pad= 1 ,bn= False) #[N, 64, 56, 56]
        self.conv3 = conv(c_in= 2*conv_dim, c_out= 4*conv_dim, k_size= 4, stride= 2, pad= 1 ,bn= False) #[N, 64, 28, 28]
        self.conv4 = conv(c_in= 4*conv_dim, c_out= 4*conv_dim, k_size= 6, stride= 4, pad= 3 ,bn= False) #[N, 64, 8, 8]
        self.fc = conv(conv_dim*4, 1, 8, 2, 0, False) #[N, 1, 1, 1]
        
    def forward(self, x):
        out = F.leaky_relu(self.conv1(x), 0.2)    
        out = F.leaky_relu(self.conv2(out), 0.2) 
        out = F.leaky_relu(self.conv3(out), 0.2) 
        out = F.leaky_relu(self.conv4(out), 0.2) 
        out = self.fc(out).squeeze()
        out = out.unsqueeze(1)
        return out