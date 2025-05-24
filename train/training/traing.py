import torch
from torch.utils.data import DataLoader, random_split
from utils import plot_loss, save_checkpoint

def train_model(model, dataloader, test_loader, criterion, optimizer, epochs, device):
    model.to(device)
    train_losses = []
    test_losses = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for r_batch, m_batch in dataloader:
            r_batch, m_batch = r_batch.to(device), m_batch.to(device)
            e_pred = model(r_batch)
            loss = criterion(r_batch, e_pred, m_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        train_losses.append(avg_loss)

        # Evaluate on test data
        model.eval()
        with torch.no_grad():
            total_test_loss = 0
            for r_batch, m_batch in test_loader:
                r_batch, m_batch = r_batch.to(device), m_batch.to(device)
                e_pred = model(r_batch)
                test_loss = criterion(r_batch, e_pred, m_batch).item()
                total_test_loss += test_loss
            test_losses.append(total_test_loss / len(test_loader))

        # Plot and save
        plot_loss(train_losses, test_losses, epoch)
        save_checkpoint(model, optimizer, epoch, f"checkpoint_epoch_{epoch}.pt")

        print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {avg_loss:.4f} - Test Loss: {test_losses[-1]:.4f}")
