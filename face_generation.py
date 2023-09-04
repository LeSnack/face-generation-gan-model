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
        
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channels)
        )
            
    def forward(self, x):
        return x + self.block(x)
class Generator(nn.Module):
    def __init__(self, nz, ngf, nc):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            spectral_norm(nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False)),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            
            spectral_norm(nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            ResidualBlock(ngf * 4),  # Adding the residual block here
            
            spectral_norm(nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            
            spectral_norm(nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            ResidualBlock(ngf),  # Adding another residual block here
            
            spectral_norm(nn.ConvTranspose2d(ngf, ngf, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 64 x 64

            spectral_norm(nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False)),
            nn.Tanh()
            # final state size. (nc) x 128 x 128
        )

    def forward(self, input):
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
    noise = torch.randn(num_faces, nz, 1, 1, device=device, dtype=torch.float32)
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


