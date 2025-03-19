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
            loss_value = segmentation_loss(x_batch, p_batch, m_batch, alpha=alpha)
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


def create_model(img_directory, pat_directory):
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
    return EncoderDecoder(in_channels=2), dataloader