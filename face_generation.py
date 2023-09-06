import torch
import torchvision
import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from torch.nn.utils import spectral_norm

class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        
        # Define the residual block with two convolutional layers
        # followed by batch normalization. The number of channels remains
        # unchanged across the block.
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channels)
        )
            
    def forward(self, x):
        # Add the input to the output of the block (residual connection)
        return x + self.block(x)

class Generator(nn.Module):
    def __init__(self, nz, ngf, nc):
        super(Generator, self).__init__()
        # Define the generator architecture
        self.main = nn.Sequential(
            # Initial convolutional transpose layer to upscale the latent vector
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # Upscale to 8x8
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # Upscale to 16x16
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # Apply residual block
            ResidualBlock(ngf * 2),
            # Upscale to 32x32
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # Upscale to final image size of 64x64 with 3 channels (RGB)
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        # Forward pass through the generator
        return self.main(input)

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

# Hyperparameters
nc = 3 # Number of channels in the training images
nz = 100 # Size of z latent vector (i.e. size of generator input)
ngf = 128 # Size of feature maps in generator
ndf = 128 # Size of feature maps in discriminator

# Load trained models
netG = Generator(nz, ngf, nc)
netG.load_state_dict(torch.load('generator.pth'))
netG = netG.to(device, dtype=torch.float32)
netG.eval()

# Generate faces
def generate_faces(netG, device, num_faces=1):
    noise = torch.randn((num_faces, nz, 1, 1), device=device, dtype=torch.float32)
    with torch.no_grad():
        faces = netG(noise).detach().cpu()
    return faces

# Display the faces
def imshow(img):
    img = img / 2 + 0.5 # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis('off')
    plt.show()

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Generate faces using a trained GAN')
parser.add_argument('--num_faces', type=int, default=5, help='number of faces to generate (default: 5)')
args = parser.parse_args()

# Generate and display faces
num_faces = args.num_faces
faces = generate_faces(netG, device, num_faces)
imshow(torchvision.utils.make_grid(faces, padding=2, normalize=True))


