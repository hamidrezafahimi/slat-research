from torch.utils.data import Dataset
from PIL import Image
import torchvision

# ================================
# Dataset Definition
# ================================
class ImagePatternDataset(Dataset):
    """
    Dataset for loading depth images and template masks.
    
    Parameters:
        depth_img_paths (list): List of file paths to the depth images.
        template_mask_paths (list): List of file paths to the template masks.
        transform (callable, optional): Optional transform to be applied on an image.
    """
    def __init__(self, depth_img_paths, template_mask_paths, transform=None):
        if len(depth_img_paths) != len(template_mask_paths):
            raise ValueError("The number of depth images and template masks must match.")
        self.depth_img_paths = depth_img_paths
        self.template_mask_paths = template_mask_paths
        self.transform = transform if transform is not None else torchvision.transforms.ToTensor()
        
    def __len__(self):
        return len(self.depth_img_paths)
    
    def __getitem__(self, idx):
        depth_img = Image.open(self.depth_img_paths[idx]).convert("L")
        template_mask = Image.open(self.template_mask_paths[idx]).convert("L")
        
        depth_img = self.transform(depth_img)
        template_mask = self.transform(template_mask)
        
        return depth_img, template_mask

