import torch
from torch.utils.data import Dataset
import pandas as pd

class RMDataLoader(Dataset):
    def __init__(self, r_file, m_file):
        """
        Load r and m data from CSV files.
        Args:
            r_file (str): Path to r.csv.
            m_file (str): Path to m.csv.
        """
        self.r_data = pd.read_csv(r_file).values
        self.m_data = pd.read_csv(m_file).values
        assert self.r_data.shape == self.m_data.shape, "r and m must have the same shape"

    def __len__(self):
        return len(self.r_data)

    def __getitem__(self, idx):
        r_row = torch.tensor(self.r_data[idx], dtype=torch.float32)
        m_row = torch.tensor(self.m_data[idx], dtype=torch.float32)
        return r_row, m_row
