import os
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image
from model import *

class ImagePatternDataset(Dataset):
    """
    Loads pairs of images:
      - X: the input image (grayscale)
      - P: the pattern image (grayscale)
    Both get transformed into tensors.
    """
    def __init__(self, image_paths, pattern_paths, transform=None, img_size=(320, 320)):
        self.image_paths = image_paths
        self.pattern_paths = pattern_paths
        self.img_size = img_size
        self.transform = transform if transform else T.ToTensor()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 1) Load images
        img_path = self.image_paths[idx]
        pat_path = self.pattern_paths[idx]

        # print(f"Loading: {img_path}, {pat_path}")  # Debugging print

        img = Image.open(img_path).convert('L')   # Grayscale
        pat = Image.open(pat_path).convert('L')   # Grayscale

        # 2) Resize both images to the same fixed size
        img = img.resize(self.img_size, Image.BILINEAR)
        pat = pat.resize(self.img_size, Image.BILINEAR)

        # 3) Transform to tensors
        x = self.transform(img)  # shape (1, H, W)
        p = self.transform(pat)  # shape (1, H, W)

        return x, p


# Specify directories
img_directory = "/home/hamid/w/viot3/data1005/full"
pat_directory = "/home/hamid/w/viot3/data1005/pattern"

# Get full file paths
image_file_paths = [os.path.join(img_directory, f) for f in os.listdir(img_directory) if f.endswith(".jpg")]
pattern_file_paths = [os.path.join(pat_directory, f) for f in os.listdir(pat_directory) if f.endswith(".jpg")]

# Ensure lists are sorted so they match correctly
image_file_paths.sort()
pattern_file_paths.sort()

# Check if lengths match
if len(image_file_paths) != len(pattern_file_paths):
    raise ValueError(f"Number of images ({len(image_file_paths)}) and patterns ({len(pattern_file_paths)}) do not match!")

# Create dataset and dataloader
dataset = ImagePatternDataset(image_file_paths, pattern_file_paths)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Build the model
model = EncoderDecoder(in_channels=2)

# Train
train_model(model, dataloader, epochs=20, alpha=0.01, lr=1e-4, device='cuda')
