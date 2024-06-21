import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import numpy as np


# Custom dataset class to load UV, depth, and PNG images
class ImageObjDataset(Dataset):
    def __init__(self, root_dir, png_dir, transform1=None, transform2=None):
        self.root_dir = root_dir  # Root directory of the original data
        self.png_dir = png_dir  # Directory containing the new point cloud images
        self.transform1 = transform1 # Transform for UV and point cloud images
        self.transform2 = transform2 # Transform for depth images
        self.data_info = self._get_data_info() # Get the dataset information

    # Helper function to gather information about the dataset
    def _get_data_info(self):
        data_info = []
        # Iterate over each folder in the root directory
        for trainset_folder in os.listdir(self.root_dir):
            trainset_path = os.path.join(self.root_dir, trainset_folder)
            if not os.path.isdir(trainset_path):
                # print(f"Skipping {trainset_path}, not a directory")
                continue

            # Iterate over each person folder in the trainset folder
            for person_folder in os.listdir(trainset_path):
                person_path = os.path.join(trainset_path, person_folder)
                if not os.path.isdir(person_path):
                    # print(f"Skipping {person_path}, not a directory")
                    continue

                # Directory containing depth maps
                dp_dir = os.path.join(person_path, 'dpmap')
                # Directory containing UV JPGs
                uv_jpg_dir = os.path.join(person_path, 'models_reg')

                # Directory containing new point cloud images
                png_trainset_path = os.path.join(self.png_dir, trainset_folder)
                png_person_path = os.path.join(png_trainset_path, person_folder, 'models_reg') # Directory containing new point cloud PNGs

                 # Check if necessary directories exist
                if not os.path.exists(dp_dir) or not os.path.exists(uv_jpg_dir) or not os.path.exists(png_person_path):
                    # print(f"Skipping {person_path}, required directory does not exist")
                    continue

                # List all depth map files
                dp_files = [f for f in os.listdir(dp_dir) if f.endswith('.png')]
                # print(f"Found {len(dp_files)} depth map files in {dp_dir}")

                # Iterate over each depth map file
                for dp_file in dp_files:
                    base_name = os.path.splitext(dp_file)[0]
                    uv_jpg_path = os.path.join(uv_jpg_dir, base_name + '.jpg')
                    png_path = os.path.join(png_person_path, base_name + '.png')  #  # Path to new point cloud PNG file
                    dp_path = os.path.join(dp_dir, dp_file)
                    
                    # Check if corresponding UV JPG and new point cloud PNG exist
                    if os.path.exists(uv_jpg_path) and os.path.exists(png_path):
                        data_info.append((uv_jpg_path, dp_path, png_path))
                    else:
                        print(f"Skipping {base_name}, UV JPG or PNG file does not exist")

        # print(f"Loaded {len(data_info)} items from dataset")
        return data_info

     # Return the length of the dataset
    def __len__(self):
        return len(self.data_info)

    # Retrieve an item from the dataset
    def __getitem__(self, idx):
        uv_path, dp_path, png_path = self.data_info[idx]
        
        # Load UV image and apply transform
        uv = Image.open(uv_path).convert("RGB")
        if self.transform1:
            uv = self.transform1(uv)

         # Load depth map image and apply transform
        dp = Image.open(dp_path).convert("L")
        if self.transform2:
            dp = self.transform2(dp)
        
        # Load new PNG image and apply transform
        png = Image.open(png_path).convert("RGB")
        if self.transform1:
            png = self.transform1(png)

        return uv, dp, png

# Function to get transformations for images
def get_transform(mode):
    if mode == 'RGB':
        return transforms.Compose([
            transforms.Resize((128, 128)),  # Resize image to 256x256
            transforms.ToTensor(),          # Convert image to tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize image
        ])
    elif mode == 'L':  # 灰度图像
        return transforms.Compose([
            transforms.Resize((128, 128)),  # Resize image to 256x256
            transforms.ToTensor(),          # Convert image to tensor
            transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize image
        ])
    else:
        raise ValueError(f"Unsupported image mode: {mode}")

# Function to get a data loader for the dataset
def get_data_loader(root_dir, png_dir, batch_size, num_workers=4):
    dataset = ImageObjDataset(
        root_dir=root_dir,
        png_dir=png_dir,
        transform1=get_transform('RGB'),   # Transform for UV and PNG images
        transform2=get_transform('L')      # Transform for depth images
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

# # 使用示例
# if __name__ == "__main__":
#     root_dir = './dataset/data'   # Root directory of original data
#     png_dir = './dataset/images'  # Root directory of newly generated PNG color images
#     batch_size = 32
#     data_loader = get_data_loader(root_dir, png_dir, batch_size)
    
#     for uv, dp, png in data_loader:
#         print(f"UV shape: {uv.shape}, DP shape: {dp.shape}, PNG shape: {png.shape}")
