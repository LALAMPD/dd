import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import time
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from tqdm import tqdm
from model_mesh5_reducetime import VAEEncoder, BilinearModel, Generator, Discriminator, QNetwork
from dataloader import get_data_loader

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
data_loader = get_data_loader(root_dir, png_dir, batch_size, num_workers=4)  # Increase num_workers

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

# Function to save the model checkpoints
def save_model(generator, discriminator, g_optimizer, d_optimizer, epoch, path="model_checkpoint.pth"):
    torch.save({
        'epoch': epoch,
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'vae_encoder_state_dict': vae_encoder.state_dict(),
        'optimizer_g_state_dict': g_optimizer.state_dict(),
        'optimizer_d_state_dict': d_optimizer.state_dict(),
        'optimizer_vae_state_dict': vae_optimizer.state_dict(),
    }, path)

# Function to save generated images and corresponding 3D point clouds
def save_image_and_obj(fake_images, epoch, output_dir='output_flow5'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save image
    image_file_path = os.path.join(output_dir, f'generated_image_epoch_{epoch}.png')
    fake_image = fake_images[0].detach().cpu().numpy().transpose(1, 2, 0)
    fake_image = (fake_image * 255).astype(np.uint8)
    image = Image.fromarray(fake_image)
    image.save(image_file_path)

# Function to save losses to a CSV file
import csv
def save_losses_to_csv(d_losses, g_losses, q_losses, filename='losses.csv'):
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Epoch', 'D Loss', 'G Loss', 'Q Loss'])
        for epoch, (d_loss, g_loss, q_loss) in enumerate(zip(d_losses, g_losses, q_losses)):
            writer.writerow([epoch, d_loss, g_loss, q_loss])

# Function to plot losses
def plot_losses(d_losses, g_losses, q_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(d_losses, label='D Loss')
    plt.plot(g_losses, label='G Loss')
    plt.plot(q_losses, label='Q Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Function Convergence')
    plt.legend()
    plt.show()

# Function to save latent vectors
def save_latent_vectors(latent_vectors, epoch, output_dir='latent_vectors'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    np.save(os.path.join(output_dir, f'latent_vectors_epoch_{epoch}.npy'), latent_vectors)

# Function to visualize latent vectors and save the plot
def visualize_latent_vectors(latent_vectors, epoch, output_dir='latent_vectors'):
    pca = PCA(n_components=2)
    latent_vectors_2d = pca.fit_transform(latent_vectors)
    plt.figure(figsize=(8, 6))
    plt.scatter(latent_vectors_2d[:, 0], latent_vectors_2d[:, 1], c='blue', label='Latent Vectors', alpha=0.5)
    plt.xlabel('PCA Dimension 1')
    plt.ylabel('PCA Dimension 2')
    plt.legend()
    plt.title(f'Latent Space Visualization at Epoch {epoch}')
    plt_path = os.path.join(output_dir, f'latent_vectors_epoch_{epoch}.png')
    plt.savefig(plt_path)
    plt.close()

# Training process
epochs = 20
for epoch in range(epochs):
    # Initialize timing accumulators
    total_data_loading_time = 0
    total_vae_time = 0
    total_sampling_time = 0
    total_d_time = 0
    total_g_time = 0

    epoch_d_loss = 0.0
    epoch_g_loss = 0.0
    epoch_q_loss = 0.0
    num_batches = 0
    latent_vectors_epoch = []

    with tqdm(total=len(data_loader), desc=f"Epoch {epoch+1}/{epochs}") as pbar:
        # Training loop for each batch
        for i, (uv_data, depth_data, real_images) in enumerate(data_loader):
            step_start_time = time.time()

            # Prepare data
            uv_data = uv_data.to(device)
            depth_data = depth_data.to(device)
            real_images = real_images.to(device)
            data_loading_time = time.time() - step_start_time
            total_data_loading_time += data_loading_time

            # VAE encoding (reparameterization trick)
            vae_start_time = time.time()
            vae_optimizer.zero_grad()
            mu, logvar = vae_encoder(uv_data, depth_data)
            std = torch.exp(0.5 * logvar)
            z = mu + std * torch.randn_like(std)
            latent_vectors_epoch.append(z.detach().cpu().numpy())
            vae_time = time.time() - vae_start_time
            total_vae_time += vae_time

            # Randomly sample continuous and discrete variables
            sample_start_time = time.time()
            continuous_c = torch.FloatTensor(uv_data.size(0), con_dim).uniform_(-1, 1).to(device) # sample from uniform distribution [-1, 1] 
            discrete_c = F.one_hot(torch.randint(0, discrete_dim, (uv_data.size(0),)), num_classes=discrete_dim).float().to(device) # sample from one-hot distribution
            c = torch.cat((continuous_c, discrete_c), dim=1).to(device)
            sampling_time = time.time() - sample_start_time
            total_sampling_time += sampling_time

            # Train discriminator and generator
            d_start_time = time.time()
            d_loss = train_discriminator(real_images, z, c)
            d_time = time.time() - d_start_time
            total_d_time += d_time

            g_start_time = time.time()
            g_loss, q_loss_value = 0, 0
            for j in range(3):   # Adjust the number of times the generator is trained
                if j == 1:       # Last time, do not retain graph
                    g_loss, q_loss_value, fake_images = train_generator(z, c)
                else:
                    g_loss, q_loss_value, _ = train_generator(z, c)
            g_time = time.time() - g_start_time
            total_g_time += g_time

            # Record loss values
            epoch_d_loss += d_loss
            epoch_g_loss += g_loss / 2  # average the generator loss over the two updates
            epoch_q_loss += q_loss_value / 2  # average the Q loss over the two updates
            num_batches += 1

            # Update progress bar
            pbar.update(1)
    
    # Record average loss values for the epoch
    if num_batches > 0:
        d_losses.append(epoch_d_loss / num_batches)
        g_losses.append(epoch_g_loss / num_batches)
        q_losses.append(epoch_q_loss / num_batches)

    # Save and visualize latent vectors for the epoch
    latent_vectors_epoch = np.concatenate(latent_vectors_epoch, axis=0)
    save_latent_vectors(latent_vectors_epoch, epoch)
    visualize_latent_vectors(latent_vectors_epoch, epoch)

    # Save generated images and model checkpoints at the end of each epoch
    save_image_and_obj(fake_images, epoch)
    save_model(generator, discriminator, g_optimizer, d_optimizer, epoch)

    # End timing
    end_time = time.time()
    elapsed_time = end_time - step_start_time
    print(f"Epoch {epoch+1}/{epochs}: Data Load Time: {total_data_loading_time:.2f}s, VAE Time: {total_vae_time:.2f}s, Sample Time: {total_sampling_time:.2f}s, D Time: {total_d_time:.2f}s, G Time: {total_g_time:.2f}s, Total Time: {elapsed_time:.2f}s")

# Save losses to a CSV file
save_losses_to_csv(d_losses, g_losses, q_losses)

# Plot losses
plot_losses(d_losses, g_losses, q_losses)
