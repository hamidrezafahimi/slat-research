import torch
from torch.utils.data import DataLoader, random_split
from dataset import RMDataLoader
from model import ENet
from loss import CombinedLoss
from train import train_model

# Hyperparameters
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
TEST_SPLIT = 0.2
LAMBDA_SMOOTH = 0.1
LAMBDA_REG = 0.1
G_MIN, G_MAX = -1.0, 1.0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset
dataset = RMDataLoader("r.csv", "m.csv")
train_size = int((1 - TEST_SPLIT) * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# Initialize model, loss, optimizer
input_dim = dataset[0][0].shape[0]
model = ENet(input_dim)
criterion = CombinedLoss(LAMBDA_SMOOTH, LAMBDA_REG, G_MIN, G_MAX)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Train model
train_model(model, train_loader, test_loader, criterion, optimizer, EPOCHS, DEVICE)
