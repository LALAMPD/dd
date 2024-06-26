import os
import numpy as np
from PIL import Image
import torch
print(torch.cuda.is_available())  # 应该返回 True
print(torch.backends.cudnn.enabled)  # 应该返回 True
print(torch.cuda.current_device())  # 应该返回 0
print(torch.cuda.get_device_name(0))  # 应该返回你的 GPU 名称
import torch.nn as nn
import torch.optim as optim
import time
import open3d as o3d
import torch.nn.functional as F
import matplotlib.pyplot as plt
from model_mesh5_reducetime import VAEEncoder, BilinearModel, Generator, Discriminator, QNetwork
from dataloader import get_data_loader

import torch

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 64

# Parameters
height, width = 128, 128 # Dimensions of UV maps and depth maps
latent_dim = 200 # Dimension of random noise z
discrete_dim = 5 # Dimension of discrete code c
con_dim = 5  # Dimension of continuous variables u
code_dim = discrete_dim + con_dim  # Dimension of latent variables c
uv_channels = 3 # Number of channels in UV maps
dp_channels = 1 # Number of channels in depth maps

# Initialize data loader
root_dir = './dataset/data'
png_dir = './dataset/images'
data_loader = get_data_loader(root_dir, png_dir, batch_size,num_workers=4)  # Increase num_workers

# Initialize models
vae_encoder = VAEEncoder(uv_channels, dp_channels, latent_dim=latent_dim).to(device)
bilinear_model = BilinearModel(latent_dim, code_dim).to(device)
generator = Generator(latent_dim, code_dim).to(device)
discriminator = Discriminator().to(device)
q_network = QNetwork(code_dim).to(device)

# Optimizers
d_optimizer = optim.Adam(discriminator.parameters(), lr=0.00001, betas=(0.5, 0.999))
g_optimizer = optim.Adam(generator.parameters(), lr=0.0001, betas=(0.5, 0.999))
q_optimizer = optim.Adam(q_network.parameters(), lr=0.0001, betas=(0.5, 0.999))
vae_optimizer = optim.Adam(vae_encoder.parameters(), lr=0.0001, betas=(0.5, 0.999))


# Loss functions
adversarial_loss = nn.BCELoss()
q_loss = nn.MSELoss()

# Loss functions
d_losses = []
g_losses = []
q_losses = []

# Training loss function for discriminator
def train_discriminator(real_images, z, c):
    # Label for real images and fake images
    valid = torch.ones(real_images.size(0), 1, device=device)
    fake = torch.zeros(real_images.size(0), 1, device=device)

    # Generate fake images from the generator using noise z and latent code c
    fake_images = generator(z, c)
    # Reset gradients for the discriminator
    d_optimizer.zero_grad()

    # Discriminator prediction for real images
    real_pred = discriminator(real_images)
    # Calculate loss for real images
    d_loss_real = adversarial_loss(real_pred, valid)
    # Discriminator prediction for fake images
    fake_pred = discriminator(fake_images.detach())
    # Calculate loss for fake images
    d_loss_fake = adversarial_loss(fake_pred, fake)

    # Total discriminator loss
    d_loss = (d_loss_real + d_loss_fake) / 2
    # Backpropagate the loss，calculating gradients
    d_loss.backward()

    # Gradient clipping to avoid exploding gradients
    torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 1.0)  
    # Update discriminator weights
    d_optimizer.step()

    return d_loss.item()

# Training loss function for generator and Q network
def train_generator(z, c):
    # Labels for valid data
    valid = torch.ones(real_images.size(0), 1, device=device)

    # Reset gradients for the generator and q-network
    g_optimizer.zero_grad()
    q_optimizer.zero_grad()

    # Generate fake images from the generator using noise z and latent code c
    fake_images = generator(z, c)

    # Discriminator prediction for fake images
    validity = discriminator(fake_images)
    # Calculate adversarial loss for generator
    g_loss = adversarial_loss(validity, valid)
    # Predict latent code c from the Q network
    predicted_c = q_network(fake_images)
    # Calculate loss for latent code prediction
    q_loss_value = q_loss(predicted_c, c)

    # Total generator loss
    g_loss_total = g_loss + q_loss_value
    # Backpropagate the loss
    g_loss_total.backward(retain_graph=True)

    # Gradient clipping for generator and Q network to avoid exploding gradients
    torch.nn.utils.clip_grad_norm_(generator.parameters(), 1.0)  
    torch.nn.utils.clip_grad_norm_(q_network.parameters(), 1.0)  
    # Update generator weights and Q network weights
    g_optimizer.step()
    q_optimizer.step()

    return g_loss.item(), q_loss_value.item(), fake_images

# Training loop
epochs = 188

# Function to save the model checkpoints
def save_model(generator, discriminator, g_optimizer, d_optimizer, epoch, i, path="model_checkpoint.pth"):
    torch.save({
        'epoch': epoch,
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'vae_encoder_state_dict': vae_encoder.state_dict(),
        'optimizer_g_state_dict': g_optimizer.state_dict(),
        'optimizer_d_state_dict': d_optimizer.state_dict(),
        'optimizer_vae_state_dict': vae_optimizer.state_dict(),
        'iteration': i
    }, path)

# Training process
for epoch in range(epochs):
    # traning a epoch
    for i, (uv_data, depth_data, real_images) in enumerate(data_loader):
        # Prepare data
        uv_data = uv_data.to(device)
        depth_data = depth_data.to(device)
        real_images = real_images.to(device)
       
        # VAE encoding (reparameterization trick)
        vae_optimizer.zero_grad()

        mu, logvar = vae_encoder(uv_data, depth_data)
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)
        print (mu.shape, std.shape)

        # Randomly sample continuous and discrete variables
        continuous_c = torch.FloatTensor(uv_data.size(0), con_dim).uniform_(-1, 1).to(device) # sample from uniform distribution [-1, 1] 
        discrete_c = F.one_hot(torch.randint(0, discrete_dim, (uv_data.size(0),)), num_classes=discrete_dim).float().to(device) # sample from one-hot distribution
        c = torch.cat((continuous_c, discrete_c), dim=1).to(device)

        # Train discriminator and generator
        d_loss = train_discriminator(real_images, z, c)

        # Train generator and Q network twice
        g_loss, q_loss_value = 0, 0
        for j in range(3):   # Adjust the number of times the generator is trained
            if j == 1:       # Last time, do not retain graph
                g_loss, q_loss_value, fake_images = train_generator(z, c)
            else:
                g_loss, q_loss_value, _ = train_generator(z, c)

        
        # Save model checkpoints and generated images and point clouds
        if i % 10 == 0:
            print(f"Epoch {epoch}, Batch {i}: [D loss: {d_loss}] [G loss: {g_loss}] [Q loss: {q_loss_value}]")
            save_model(generator, discriminator, g_optimizer, d_optimizer, epoch, i)

           