import torch
import torch.nn as nn
import torch.nn.functional as F

# Set random seed for reproducibility
torch.manual_seed(0)

# Define Residual Block
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

class VAEEncoder(nn.Module):
    def __init__(self, uv_channels, dp_channels, latent_dim):
        super(VAEEncoder, self).__init__()
        self.uv_conv1 = nn.Conv2d(uv_channels, 32, kernel_size=3, stride=2, padding=1)
        self.uv_conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.uv_conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)

        self.dp_conv1 = nn.Conv2d(dp_channels, 32, kernel_size=3, stride=2, padding=1)
        self.dp_conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.dp_conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)

        self.fc_mu = nn.Linear(256 * 16 * 16, latent_dim)
        self.fc_logvar = nn.Linear(256 * 16 * 16, latent_dim)

    def forward(self, uv, dp):
        uv_h = F.relu(self.uv_conv1(uv))
        uv_h = F.relu(self.uv_conv2(uv_h))
        uv_h = F.relu(self.uv_conv3(uv_h))
        
        dp_h = F.relu(self.dp_conv1(dp))
        dp_h = F.relu(self.dp_conv2(dp_h))
        dp_h = F.relu(self.dp_conv3(dp_h))
        
        h = torch.cat((uv_h, dp_h), dim=1) 
        h = h.view(h.size(0), -1)

        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

class BilinearModel(nn.Module):
    def __init__(self, z_dim, c_dim):
        super(BilinearModel, self).__init__()
        self.z_fc = nn.Linear(z_dim, z_dim)
        self.c_fc = nn.Linear(c_dim, z_dim)
        self.hidden_fc = nn.Linear(z_dim + z_dim + c_dim, z_dim + c_dim)

    def forward(self, z, c):
        z_mapped = self.z_fc(z)
        c_mapped = self.c_fc(c)
        z_prime = z_mapped * c_mapped
        combined = torch.cat([z_prime, z, c], dim=1)
        combined = F.relu(self.hidden_fc(combined))
        return combined

class Generator(nn.Module):
    def __init__(self, latent_dim, code_dim):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.code_dim = code_dim
        self.bilinear = BilinearModel(latent_dim, code_dim)
        self.init_size = 8
        self.l1 = nn.Sequential(nn.Linear(latent_dim + code_dim, 128 * self.init_size * self.init_size))
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            ResidualBlock(64),  # 添加残差块
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            ResidualBlock(32),  # 添加残差块
            nn.Upsample(scale_factor=2),
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            ResidualBlock(16),  # 添加残差块
            nn.Upsample(scale_factor=2),
            nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, z, c):
        combined = self.bilinear(z, c)
        out = self.l1(combined)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.fc1 = nn.Linear(128 * 8 * 8, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.fc1(out)
        validity = self.sigmoid(validity)
        return validity

class QNetwork(nn.Module):
    def __init__(self, code_dim):
        super(QNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.fc1 = nn.Linear(128 * 8 * 8, 1024)
        self.fc2 = nn.Linear(1024, code_dim)

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        out = F.relu(self.fc1(out))
        code = self.fc2(out)
        return code

################### test ###################

# # Parameters
# batch_size = 8
# latent_dim = 100
# code_dim = 10
# uv_channels = 3
# dp_channels = 1
# image_size = (uv_channels, 128, 128)
# depth_size = (dp_channels, 128, 128)

# # Instantiate models
# vae_encoder = VAEEncoder(uv_channels, dp_channels, latent_dim)
# bilinear_model = BilinearModel(latent_dim, code_dim)
# generator = Generator(latent_dim, code_dim)
# discriminator = Discriminator()
# qnetwork = QNetwork(code_dim)

# # Test VAEEncoder
# uv = torch.randn(batch_size, *image_size) # Random UV image
# depth = torch.randn(batch_size, *depth_size) # Random depth image
# mu, logvar = vae_encoder(uv, depth)
# print(f"VAE Encoder output shapes: mu={mu.shape}, logvar={logvar.shape}")

# # Test BilinearModel
# z = torch.randn(batch_size, latent_dim)
# c = torch.randn(batch_size, code_dim)
# combined = bilinear_model(z, c)
# print(f"Bilinear Model output shape: {combined.shape}")

# # Test Generator
# generated_image = generator(z, c)
# print(f"Generated image shape: {generated_image.shape}")

# # Test Discriminator
# validity = discriminator(generated_image)
# print(f"Discriminator output shape: {validity.shape}")

# # Test QNetwork
# code = qnetwork(generated_image)
# print(f"QNetwork output shape: {code.shape}")

# # Check output shapes to ensure they match expectations
# assert mu.shape == (batch_size, latent_dim), f"VAE Encoder mu shape mismatch: {mu.shape}"
# assert logvar.shape == (batch_size, latent_dim), f"VAE Encoder logvar shape mismatch: {logvar.shape}"
# assert combined.shape == (batch_size, latent_dim + code_dim), f"Bilinear Model shape mismatch: {combined.shape}"
# assert generated_image.shape == (batch_size, 3, 128, 128), f"Generated image shape mismatch: {generated_image.shape}"
# assert validity.shape == (batch_size, 1), f"Discriminator output shape mismatch: {validity.shape}"
# assert code.shape == (batch_size, code_dim), f"QNetwork output shape mismatch: {code.shape}"

# print("All tests passed!")
