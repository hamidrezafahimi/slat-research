import torch
import matplotlib.pyplot as plt

def save_checkpoint(model, optimizer, epoch, filename):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
    }, filename)

def plot_loss(train_losses, test_losses, epoch):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(test_losses, label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Loss after {epoch+1} epochs")
    plt.legend()
    plt.grid(True)
    plt.pause(0.01)
    plt.show()
