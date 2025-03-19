from torch.utils.data import Dataset
from PIL import Image
import torchvision

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
        self.transform = transform if transform else torchvision.transforms.ToTensor()

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