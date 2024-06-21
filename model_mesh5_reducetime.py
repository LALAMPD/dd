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
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channels)
        )
 
    def forward(self, x):
        return x + self.block(x)
 
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
        # print(f"Concatenated UV and DP shape: {x.shape}")
 
        # 第一阶段：大步幅卷积
        x = F.relu(self.conv1(x))
        # print(f"After conv1 shape: {x.shape}")
        x = F.relu(self.conv2(x))
        # print(f"After conv2 shape: {x.shape}")
 
        # 提取中心区域
        center_x = x[:, :, 8:24, 8:24].clone()  # 提取中心 16x16 区域，使用clone防止就地操作
        # print(f"Center region shape: {center_x.shape}")
 
        # 第二阶段：小步幅卷积
        center_x = F.relu(self.center_conv1(center_x))
        # print(f"After center_conv1 shape: {center_x.shape}")
        center_x = F.relu(self.center_conv2(center_x))
        # print(f"After center_conv2 shape: {center_x.shape}")
 
        # 将处理后的中心区域和外围区域拼接在一起
        x_clone = x.clone()  # Clone x before modifying it
        x_clone[:, :, 8:24, 8:24] = center_x
        # x[:, :, 8:24, 8:24] = center_x
        # print(f"Combined center and outer region shape: {x.shape}")
 
        # 全局平均池化
        x = self.global_pool(x)
        # print(f"After global_pool shape: {x.shape}")
        x = x.view(x.size(0), -1)
        # print(f"After view shape: {x.shape}")
 
        mu = self.fc_mu(x)
        # print(f"mu shape: {mu.shape}")
        logvar = self.fc_logvar(x)
        # print(f"logvar shape: {logvar.shape}")
 
        return mu, logvar
 
class BilinearModel(nn.Module):
    def __init__(self, z_dim, c_dim):
        super(BilinearModel, self).__init__()
        self.z_fc = nn.Linear(z_dim, z_dim)
        self.c_fc = nn.Linear(c_dim, z_dim)
        self.hidden_fc = nn.Linear(z_dim + z_dim + c_dim, z_dim + c_dim)
 
    def forward(self, z, c):
        z_mapped = self.z_fc(z)
        # print(f"z_mapped shape: {z_mapped.shape}")
        c_mapped = self.c_fc(c)
        # print(f"c_mapped shape: {c_mapped.shape}")
        z_prime = z_mapped * c_mapped
        # print(f"z_prime shape: {z_prime.shape}")
        combined = torch.cat([z_prime, z, c], dim=1)
        # print(f"Combined shape: {combined.shape}")
        combined = F.relu(self.hidden_fc(combined))
        # print(f"After hidden_fc shape: {combined.shape}")
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
            nn.LeakyReLU(0.2, inplace=False),
            ResidualBlock(64),  # 添加残差块
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32, 0.8),
            nn.LeakyReLU(0.2, inplace=False),
            ResidualBlock(32),  # 添加残差块
            nn.Upsample(scale_factor=2),
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16, 0.8),
            nn.LeakyReLU(0.2, inplace=False),
            ResidualBlock(16),  # 添加残差块
            nn.Upsample(scale_factor=2),
            nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )
 
    def forward(self, z, c):
        combined = self.bilinear(z, c)
        # print(f"Combined shape after bilinear: {combined.shape}")
        out = self.l1(combined)
        # print(f"After l1 shape: {out.shape}")
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        # print(f"After view shape: {out.shape}")
        img = self.conv_blocks(out)
        # print(f"Generated image shape: {img.shape}")
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
        # print(f"After model shape: {out.shape}")
        out = out.view(out.shape[0], -1)
        # print(f"After view shape: {out.shape}")
        validity = self.fc1(out)
        # print(f"After fc1 shape: {validity.shape}")
        validity = self.sigmoid(validity)
        # print(f"After sigmoid shape: {validity.shape}")
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
        # print(f"After model shape: {out.shape}")
        out = out.view(out.shape[0], -1)
        # print(f"After view shape: {out.shape}")
        out = F.relu(self.fc1(out))
        # print(f"After fc1 shape: {out.shape}")
        code = self.fc2(out)
        # print(f"After fc2 shape: {code.shape}")
        return code
 
# ################### test ###################
 
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