import torch
import torchvision
import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, nz, ngf, nc):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 16),
            nn.ReLU(True),
            # state size. (ngf*16) x 4 x 4
            nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 8 x 8
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 16 x 16
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 32 x 32
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 64 x 64
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 128 x 128
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


