import torch
from torch.utils.data import Dataset

# ============================================
# datasets for toydata
# ============================================

class toydataset(Dataset):
    """Dataset for toydata.
    
    Parameters
    ----------
    X : torch.Tensor
        Data samples. Shape (num_samples, d).
    C : torch.Tensor
        Lables of data samples. Shape (num_samples, K).
    """
    
    def __init__(self, X, C):
        self.X = X
        self.C = C

    def __len__(self):
        return self.X.size()[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = {'inputs': self.X[idx, :], 'labels': self.C[idx, :]}

        return sample