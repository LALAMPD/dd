import torch
import torch.nn as nn
import torch.nn.functional as F

# Set random seed for reproducibility
torch.manual_seed(0)
###生成器变为线性
class VAEEncoder(nn.Module):
    def __init__(self, uv_channels, dp_channels, latent_dim):
        super(VAEEncoder, self).__init__()
        combined_channels = uv_channels + dp_channels

        # 第一阶段：整个图像的卷积，使用大步幅
        self.conv1 = nn.Conv2d(combined_channels, 32, kernel_size=4, stride=2, padding=1)  # 输出尺寸 64x64
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)  # 输出尺寸 32x32

        # 第二阶段：中心区域的卷积，使用小步幅
        self.center_conv1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)  # 输出尺寸 32x32
        self.center_conv2 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)  # 输出尺寸 32x32

        # 全局平均池化
        self.global_pool = nn.AdaptiveAvgPool2d((8, 8))

        self.fc_mu = nn.Linear(64 * 8 * 8, latent_dim)
        self.fc_logvar = nn.Linear(64 * 8 * 8, latent_dim)

    def forward(self, uv, dp):
        x = torch.cat((uv, dp), dim=1)  # 合并 UV 和 DP 通道

        # 第一阶段：大步幅卷积
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        # 提取中心区域
        center_x = x[:, :, 8:24, 8:24].clone()  # 提取中心 16x16 区域，使用clone防止就地操作

        # 第二阶段：小步幅卷积
        center_x = F.relu(self.center_conv1(center_x))
        center_x = F.relu(self.center_conv2(center_x))

        # 将处理后的中心区域和外围区域拼接在一起
        x_clone = x.clone()  # Clone x before modifying it
        x_clone[:, :, 8:24, 8:24] = center_x

        # 全局平均池化
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)

        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)

        return mu, logvar

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

        self.fc_blocks = nn.Sequential(
            nn.Linear(128 * self.init_size * self.init_size, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Linear(1024, 2048),
            nn.BatchNorm1d(2048),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Linear(2048, 3 * 128 * 128),  # Changing the output to 128x128x3
            nn.Tanh()
        )

    def forward(self, z, c):
        combined = self.bilinear(z, c)
        out = self.l1(combined)
        out = out.view(out.shape[0], -1)  # Flatten the tensor
        img = self.fc_blocks(out)
        img = img.view(out.shape[0], 3, 128, 128)  # Reshape back to image format (3x128x128)
        return img

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=False)
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
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=False)
        )
        self.fc1 = nn.Linear(128 * 8 * 8, 1024)
        self.fc2 = nn.Linear(1024, code_dim)

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        out = F.relu(self.fc1(out))
        code = self.fc2(out)
        return code

# # ################### test ###################
 
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