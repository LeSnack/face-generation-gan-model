import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.nn.utils import spectral_norm
import matplotlib.pyplot as plt
import numpy as np

# Move model to GPU if available
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

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
        # Add the input to the output of the block (residual connection)
        return x + self.block(x)

class Generator(nn.Module):
    def __init__(self, nz, ngf, nc):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # Initial size: nz x 1 x 1
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # Size: (ngf*4 x 4 x 4)
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # Size: (ngf*4 x 8 x 8)
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # Apply residual
            ResidualBlock(ngf * 2),
            # Size: (ngf*2 x 16 x 16)
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # Size: (ngf x 32 x 32)
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # Final size: (nc x 64 x 64)
        )

    def forward(self, input):
        return self.main(input)

class Discriminator(nn.Module):
    def __init__(self, nc, ndf):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            ResidualBlock(ndf * 4),

            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

manual_seed = 999
torch.manual_seed(manual_seed)
torch.use_deterministic_algorithms(True) # Needed for reproducible results

# Custom weights initialization called on ``netG`` and ``netD``
# From the paper the weights should be initialised from a Normal
# Distribution with mean = 0 and S.D. = 0.2
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def main():
    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.CenterCrop((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # Load CelebA dataset
    dataset = datasets.CelebA(root='./data', split='train', transform=transform, download=True)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=6)

    # Hyperparameters
    nc = 3 # Number of channels in the training images
    nz = 100 # Size of z latent vector (i.e. size of generator input)
    ngf = 128 # Size of feature maps in generator
    ndf = 128 # Size of feature maps in discriminator

    # Create the Generator and Discriminator
    netG = Generator(nz, ngf, nc).to(device)
    netD = Discriminator(nc, ndf).to(device)

    # Apply weights to generator
    netG.apply(weights_init)
    netD.apply(weights_init)

    # Print the models
    print(netG)
    print(netD)

    # Loss function
    criterion = nn.BCELoss()

    # Optimizers
    lr_D = 0.0003
    lr_G = 0.0002
    beta1 = 0.5
    optimizerD = optim.Adam(netD.parameters(), lr=lr_D, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr_G, betas=(beta1, 0.999))
    
    # Learning Rate Schedulers
    schedulerD = optim.lr_scheduler.StepLR(optimizerD, step_size=30, gamma=0.9)  # Decays every 30 epochs
    schedulerG = optim.lr_scheduler.StepLR(optimizerG, step_size=30, gamma=0.9)

    # Labels for real and fake images with label smoothing
    real_label = 0.9
    fake_label = 0.1

    # Training loop
    num_epochs = 75 # Number of epochs
    G_losses = []
    D_losses = []
    # Initialize lists to track accuracies
    real_accuracies = []
    fake_accuracies = []
    iters = 0

    # For periodic visualization
    fixed_noise = torch.randn((40, nz, 1, 1), device=device, dtype=torch.float32)

    print("Starting Training Loop...")
    for epoch in range(1, num_epochs + 1):
        
        for i, data in enumerate(dataloader, 0):
            
            ############################
            # (1) Update D network
            ############################
            ## Train with all-real batch
            netD.zero_grad()
            # Format batch
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, device=device, dtype=torch.float32)
            # Forward pass real batch through D
            output = netD(real_cpu).view(-1)
            # Calculate loss on all-real batch
            errD_real = criterion(output, label)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, nz, 1, 1, device=device, dtype=torch.float32)
            # Generate fake image batch with G
            fake = netG(noise)
            label.fill_(fake_label)
            # Classify all fake batch with D
            output = netD(fake.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label)
            # Calculate the gradients for this batch
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # Add the gradients from the all-real and all-fake batches
            errD = errD_real + errD_fake
            # Update D
            optimizerD.step()

            # Calculate accuracies
            real_accuracy = ((output > 0.5).float() == label).float().mean().item()
            fake_accuracy = ((output < 0.5).float() == label).float().mean().item()

            # Update the lists with the calculated accuracies
            real_accuracies.append(real_accuracy)
            fake_accuracies.append(fake_accuracy)

            ############################
            # (2) Update G network
            ############################
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = netD(fake).view(-1)
            # Calculate G's loss based on this output
            errG = criterion(output, label)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            optimizerG.step()

            # Output training stats
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                    % (epoch, num_epochs, i, len(dataloader),
                        errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            iters += 1
        
        # Update learning rate at the end of each epoch
        schedulerD.step()
        schedulerG.step()

        # Visualize after every 5 epochs
        if epoch % 5 == 0:
            netG.eval()  # Set to evaluation mode
            with torch.no_grad():
                fake_images = netG(fixed_noise).detach().cpu()
            plt.figure(figsize=(8, 8))
            plt.axis("off")
            plt.imshow(np.transpose(torchvision.utils.make_grid(fake_images, padding=2, normalize=True), (1, 2, 0)))
            plt.show()
            
            netG.train()  # Set back to training mode

    # Save the models
    torch.save(netG.state_dict(), 'generator.pth')
    torch.save(netD.state_dict(), 'discriminator.pth')

    print("Training complete!")

    # Plot the training losses
    plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses,label="G")
    plt.plot(D_losses,label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    # Plot the training accuracies
    plt.figure(figsize=(10,5))
    plt.title("Real vs. Fake Accuracies During Training")
    plt.plot(real_accuracies, label="Real Accuracy")
    plt.plot(fake_accuracies, label="Fake Accuracy")
    plt.xlabel("iterations")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()