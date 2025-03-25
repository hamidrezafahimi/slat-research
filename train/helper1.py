import torch
import torch.optim as optim
from imageDataset import ImagePatternDataset
from model import *
from torch.utils.data import DataLoader
import os

def train_model(model, dataloader, epochs=10, alpha=0.01, lr=1e-4, device='cuda', checkpoint_interval=5, checkpoint_dir='./checkpoints'):
    """
    model             -> instance of EncoderDecoder
    dataloader        -> yields (image_batch, pattern_batch)
    epochs            -> number of training epochs
    alpha             -> hyperparameter for binary penalty
    lr                -> learning rate
    device            -> 'cuda' or 'cpu'
    checkpoint_interval -> save model every N epochs
    checkpoint_dir    -> directory to save the checkpoint files
    """
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Ensure the checkpoint directory exists
    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(epochs):
        epoch_loss = 0.0
        model.train()
        
        for batch_idx, (x_batch, p_batch) in enumerate(dataloader):
            x_batch = x_batch.to(device)
            p_batch = p_batch.to(device)

            optimizer.zero_grad()
            m_batch = model(x_batch, p_batch)   # forward pass
            loss_value = segmentation_loss(x_batch, p_batch, m_batch)
            loss_value.backward()               # backprop
            optimizer.step()                    # update parameters

            epoch_loss += loss_value.item()
        
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")
        
        # Save checkpoint if needed
        if (epoch + 1) % checkpoint_interval == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            print(f"Checkpoint saved at {checkpoint_path}")


# ================================
# Model Creation Function
# ================================
def create_model(depth_img_directory, template_mask_directory, batch_size=4, transform=None):
    """
    depth_img_directory    -> directory containing depth images (assumed .jpg)
    template_mask_directory-> directory containing template masks (assumed .jpg)
    Returns:
        model: an instance of UNet with in_channels=1 and n_classes=1.
        dataloader: DataLoader for the paired dataset.
    """
    # Get full file paths for depth images and template masks
    depth_img_paths = [os.path.join(depth_img_directory, f) for f in os.listdir(depth_img_directory) if f.endswith(".jpg")]
    template_mask_paths = [os.path.join(template_mask_directory, f) for f in os.listdir(template_mask_directory) if f.endswith(".jpg")]

    # Ensure lists are sorted so that pairs match correctly
    depth_img_paths.sort()
    template_mask_paths.sort()

    if len(depth_img_paths) != len(template_mask_paths):
        raise ValueError(f"Number of depth images ({len(depth_img_paths)}) and template masks ({len(template_mask_paths)}) do not match!")
    
    dataset = ImagePatternDataset(depth_img_paths, template_mask_paths, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Build the model: UNet expects one channel input and outputs one channel threshold map.
    model = UNet(n_channels=1, n_classes=1)
    return model, dataloader


# ================================
# Training Function
# ================================
def train_model(model, dataloader, epochs=10, alpha=0.01, lr=1e-4, device='cuda', checkpoint_interval=5, checkpoint_dir='./checkpoints'):
    """
    model             -> instance of UNet (or similar image-to-image model)
    dataloader        -> yields (depth_image_batch, template_mask_batch)
    epochs            -> number of training epochs
    alpha             -> hyperparameter for binary penalty
    lr                -> learning rate
    device            -> 'cuda' or 'cpu'
    checkpoint_interval -> save model every N epochs
    checkpoint_dir    -> directory to save the checkpoint files
    """
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(epochs):
        epoch_loss = 0.0
        model.train()
        
        for batch_idx, (x_batch, template_mask) in enumerate(dataloader):
            x_batch = x_batch.to(device)
            template_mask = template_mask.to(device)

            optimizer.zero_grad()
            # Forward pass: produce threshold pattern from depth image
            th_pattern = model(x_batch)
            
            # Compute loss between the masked image (using th_pattern) and the template mask
            loss_value = segmentation_loss(x_batch, template_mask, th_pattern)
            loss_value.backward()
            optimizer.step()

            epoch_loss += loss_value.item()
        
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")
        
        # Save checkpoint if needed
        if (epoch + 1) % checkpoint_interval == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            print(f"Checkpoint saved at {checkpoint_path}")