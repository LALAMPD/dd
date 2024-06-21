import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from tqdm import tqdm
from PIL import Image
import psutil
import pynvml
from torch.utils.data import TensorDataset, DataLoader
from model_mesh5_reducetime import VAEEncoder, BilinearModel, Generator, Discriminator, QNetwork
from dataloader import get_data_loader

# Initialize NVML
pynvml.nvmlInit()

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Enable cuDNN benchmark for performance optimization
torch.backends.cudnn.benchmark = True

# Parameters
batch_size = 64
height, width = 128, 128  # Dimensions of UV maps and depth maps
latent_dim = 200  # Dimension of random noise z
discrete_dim = 5  # Dimension of discrete code c
con_dim = 5  # Dimension of continuous variables u
code_dim = discrete_dim + con_dim  # Dimension of latent variables c
uv_channels = 3  # Number of channels in UV maps
dp_channels = 1  # Number of channels in depth maps

# Initialize data loader
root_dir = './dataset/data'
png_dir = './dataset/images'

# Load all data into memory with progress bar
def load_data_to_memory(root_dir, png_dir):
    data_loader = get_data_loader(root_dir, png_dir, batch_size, num_workers=16)
    all_uv_data = []
    all_depth_data = []
    all_real_images = []

    total_batches = len(data_loader)
    with tqdm(total=total_batches, desc="Loading data to memory") as pbar:
        for uv_data, depth_data, real_images in data_loader:
            all_uv_data.append(uv_data)
            all_depth_data.append(depth_data)
            all_real_images.append(real_images)
            pbar.update(1)

    all_uv_data = torch.cat(all_uv_data)
    all_depth_data = torch.cat(all_depth_data)
    all_real_images = torch.cat(all_real_images)

    return all_uv_data, all_depth_data, all_real_images

# Load data
if not os.path.exists('preloaded_data.npz'):
    all_uv_data, all_depth_data, all_real_images = load_data_to_memory(root_dir, png_dir)
    np.savez('preloaded_data.npz', uv_data=all_uv_data.numpy(), depth_data=all_depth_data.numpy(), real_images=all_real_images.numpy())
else:
    loaded_data = np.load('preloaded_data.npz')
    all_uv_data = torch.tensor(loaded_data['uv_data'])
    all_depth_data = torch.tensor(loaded_data['depth_data'])
    all_real_images = torch.tensor(loaded_data['real_images'])

# Verify data loading
print(f"UV data shape: {all_uv_data.shape}, type: {all_uv_data.dtype}")
print(f"Depth data shape: {all_depth_data.shape}, type: {all_depth_data.dtype}")
print(f"Real images shape: {all_real_images.shape}, type: {all_real_images.dtype}")

# Create TensorDataset and DataLoader
dataset = TensorDataset(all_uv_data, all_depth_data, all_real_images)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

# Monitor memory usage
def monitor_memory():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / (1024 ** 2)  # Convert to MB

print(f"Memory usage after loading data: {monitor_memory()} MB")

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
adversarial_loss = nn.BCELoss().to(device)
q_loss = nn.MSELoss().to(device)

# Loss functions
d_losses = []
g_losses = []
q_losses = []
g_weight_norms = []
d_weight_norms = []

# Function to calculate the weight norm
def calculate_weight_norm(model):
    total_norm = 0
    for param in model.parameters():
        param_norm = param.data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm

# Function to get GPU utilization
def get_gpu_utilization():
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
    memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return utilization.gpu, utilization.memory, memory_info.used / memory_info.total

# Function to get CPU utilization
def get_cpu_utilization():
    cpu_usage = psutil.cpu_percent(interval=1)
    memory_info = psutil.virtual_memory()
    return cpu_usage, memory_info.percent

# Training loss function for discriminator
def train_discriminator(real_images, z, c):
    real_images = real_images.to(device)
    z = z.to(device)
    c = c.to(device)

    # Label for real images and fake images
    valid = torch.ones(real_images.size(0), 1, device=device)
    fake = torch.zeros(real_images.size(0), 1, device=device)

    # Generate fake images from the generator using noise z and latent code c
    fake_images = generator(z, c).to(device)

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
    z = z.to(device)
    c = c.to(device)

    # Labels for valid data
    valid = torch.ones(z.size(0), 1, device=device)

    # Reset gradients for the generator and q-network
    g_optimizer.zero_grad()
    q_optimizer.zero_grad()

    # Generate fake images from the generator using noise z and latent code c
    fake_images = generator(z, c).to(device)

    # Discriminator prediction for fake images
    validity = discriminator(fake_images)
    # Calculate adversarial loss for generator
    g_loss = adversarial_loss(validity, valid)
    # Predict latent code c from the Q network
    predicted_c = q_network(fake_images).to(device)
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

# Function to plot weight norms
def plot_weight_norms(g_weight_norms, d_weight_norms):
    plt.figure(figsize=(10, 5))
    plt.plot(g_weight_norms, label='Generator Weight Norm')
    plt.plot(d_weight_norms, label='Discriminator Weight Norm')
    plt.xlabel('Epoch')
    plt.ylabel('Weight Norm')
    plt.title('Weight Norms Over Epochs')
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
epochs = 1000

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
            data_start_time = time.time()
            uv_data = uv_data.to(device)
            depth_data = depth_data.to(device)
            real_images = real_images.to(device)
            data_loading_time = time.time() - data_start_time
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
            continuous_c = torch.FloatTensor(uv_data.size(0), con_dim).uniform_(-1, 1).to(device)
            discrete_c = F.one_hot(torch.randint(0, discrete_dim, (uv_data.size(0),)), num_classes=discrete_dim).float().to(device)
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
            for j in range(3):
                if j == 1:
                    g_loss, q_loss_value, fake_images = train_generator(z, c)
                else:
                    g_loss, q_loss_value, _ = train_generator(z, c)
            g_time = time.time() - g_start_time
            total_g_time += g_time

            # Record loss values
            epoch_d_loss += d_loss
            epoch_g_loss += g_loss / 3
            epoch_q_loss += q_loss_value / 3
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

    # Calculate and record weight norms
    g_weight_norms.append(calculate_weight_norm(generator))
    d_weight_norms.append(calculate_weight_norm(discriminator))

    # End timing
    end_time = time.time()
    elapsed_time = end_time - step_start_time
    print(f"Epoch {epoch+1}/{epochs}: Data Load Time: {total_data_loading_time:.2f}s, VAE Time: {total_vae_time:.2f}s, Sample Time: {total_sampling_time:.2f}s, D Time: {total_d_time:.2f}s, G Time: {total_g_time:.2f}s, Total Time: {elapsed_time:.2f}s")
    print(f"Epoch {epoch+1}/{epochs}: D Loss: {d_losses[-1]}, G Loss: {g_losses[-1]}, Q Loss: {q_losses[-1]}")

    # Log GPU and CPU utilization
    gpu_utilization, gpu_memory_utilization, gpu_memory_used = get_gpu_utilization()
    cpu_utilization, memory_utilization = get_cpu_utilization()
    print(f"GPU Utilization: {gpu_utilization}%, GPU Memory Utilization: {gpu_memory_utilization}%, GPU Memory Used: {gpu_memory_used*100:.2f}%")
    print(f"CPU Utilization: {cpu_utilization}%, Memory Utilization: {memory_utilization}%")


# Save losses to a CSV file
save_losses_to_csv(d_losses, g_losses, q_losses)

# Plot losses
plot_losses(d_losses, g_losses, q_losses)

# Plot weight norms
plot_weight_norms(g_weight_norms, d_weight_norms)

# Shutdown NVML
pynvml.nvmlShutdown()
