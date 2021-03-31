import numpy as np
import torch

# ============================================
# 1D donut toydata
# ============================================

def donut_1D(N=1500, V=1500, binary=True):
    """Generate toydataset of 1D donut.
    
    Parameters
    ----------
    N : int, optional
        Number of training samples. Default is N = 1500.
    V : int, optional
        Number of validation samples. Default is V = 1500.
    binary : bool, optional
        Create labels as for binary classification, otherwise as for general 
        classification. Default is binary = True.
        
    Returns
    -------
    X_train : torch.Tensor
        Training data. Shape (N, 2). 
    C_train : torch.Tensor
        Lables for training data. Shape (N, K).
    X_val : torch.Tensor
        Validation data. Shape (V, 2).
    C_val : torch.Tensor
        Lables for validation data. Shape (V, K).
    d : int
        Dimension of data space. Returns d = 2.
    K : int
        Number of classes. Returns K = 1 if treated as binary classification, 
        and K = 2 if treated as general classification.
    """
    d = 2
    
    X = torch.cat((8 * torch.rand(N + V, 1) - 4, torch.randn(N + V, 1)), 1)
    R = torch.abs(X[:, 0])
    ind_class0 = (R < 1.0).nonzero()[:, 0]
    ind_class1 = (R >= 1.0).nonzero()[:, 0]
    X.add_(0.5 * torch.randn_like(X))        # add some noise to the data
    
    return label(d, X, ind_class0, ind_class1, N, V, binary)

# ============================================
# 2D donut toydata
# ============================================

def donut_2D(N=1500, V=1500, binary=True):
    """Generate toydataset of 2D donut.
    
    Parameters
    ----------
    N : int, optional
        Number of training samples. Default is N = 1500.
    V : int, optional
        Number of validation samples. Default is V = 1500.
    binary : bool, optional
        Create labels as for binary classification, otherwise as for general 
        classification. Default is binary = True.
        
    Returns
    -------
    X_train : torch.Tensor
        Training data. Shape (N, 2). 
    C_train : torch.Tensor
        Lables for training data. Shape (N, K).
    X_val : torch.Tensor
        Validation data. Shape (V, 2).
    C_val : torch.Tensor
        Lables for validation data. Shape (V, K).
    d : int
        Dimension of data space. Returns d = 2.
    K : int
        Number of classes. Returns K = 1 if treated as binary classification, 
        and K = 2 if treated as general classification.
    """
    d = 2
    
    X = torch.randn(N + V, d)
    R = torch.sqrt(X[:, 0] ** 2 + X[:, 1] ** 2)
    ind_class0 = (R < 1.0).nonzero()[:, 0]
    ind_class1 = (R >= 1.0).nonzero()[:, 0]
    X = X * 1.2                              # rescale data
    X.add_(0.2 * torch.randn_like(X))        # add some noise to the data
    
    return label(d, X, ind_class0, ind_class1, N, V, binary)

# ============================================
# Checkerboard toydata
# ============================================

def squares(N=1500, V=1500, binary=True):
    """Generate toydataset of checkerboard, i.e. squares.
    
    Parameters
    ----------
    N : int, optional
        Number of training samples. Default is N = 1500.
    V : int, optional
        Number of validation samples. Default is V = 1500.
    binary : bool, optional
        Create labels as for binary classification, otherwise as for general 
        classification. Default is binary = True.
        
    Returns
    -------
    X_train : torch.Tensor
        Training data. Shape (N, 2). 
    C_train : torch.Tensor
        Lables for training data. Shape (N, K).
    X_val : torch.Tensor
        Validation data. Shape (V, 2).
    C_val : torch.Tensor
        Lables for validation data. Shape (V, K).
    d : int
        Dimension of data space. Returns d = 2.
    K : int
        Number of classes. Returns K = 1 if treated as binary classification, 
        and K = 2 if treated as general classification.
    """
    d = 2
    
    X = 8 * torch.rand(N + V, d) - 4
    ind_class0 = (X[:, 0] * X[:, 1] > 0).nonzero()[:, 0]
    ind_class1 = (X[:, 0] * X[:, 1] <= 0).nonzero()[:, 0]
    X.add_(0.3 * torch.randn_like(X))        # add some noise to the data
    
    return label(d, X, ind_class0, ind_class1, N, V, binary)

# ============================================
# Spiral toydata
# ============================================

def spiral(N=1500, V=1500, binary=True): 
    """Generate toydataset of spiral.
    
    Parameters
    ----------
    N : int, optional
        Number of training samples. Default is N = 1500.
    V : int, optional
        Number of validation samples. Default is V = 1500.
    binary : bool, optional
        Create labels as for binary classification, otherwise as for general 
        classification. Default is binary = True.
        
    Returns
    -------
    X_train : torch.Tensor
        Training data. Shape (N, 2). 
    C_train : torch.Tensor
        Lables for training data. Shape (N, K).
    X_val : torch.Tensor
        Validation data. Shape (V, 2).
    C_val : torch.Tensor
        Lables for validation data. Shape (V, K).
    d : int
        Dimension of data space. Returns d = 2.
    K : int
        Number of classes. Returns K = 1 if treated as binary classification, 
        and K = 2 if treated as general classification.
    """
    d = 2
    
    n = (N + V + 1) // 2                            
    phi = torch.sqrt(torch.rand(n, 1)) * 2 * np.pi 

    r0 = 2 * phi + np.pi
    X0 = torch.cat((torch.cos(phi) * r0, torch.sin(phi) * r0), 1)
    X0.add_(torch.randn(n, 2))                 # add some noise to the data
    C0 = torch.zeros((n, 1))

    r1 = -2 * phi - np.pi
    X1 = torch.cat((torch.cos(phi) * r1, torch.sin(phi) * r1), 1)
    X1.add_(torch.randn(n, 2))                 # add some noise to the data
    C1 = torch.ones((n, 1))
    
    X = torch.cat((X0, X1), 0)
    C = torch.cat((C0, C1), 0)
    
    perm = np.random.permutation(2 * n)         # permute data
    X = (X[perm])[:N + V, :]
    X = X / 5                                   # rescale data
    C = (C[perm])[:N + V, :]
        
    ind_class0 = (C == 0.0).nonzero()[:, 0]
    ind_class1 = (C == 1.0).nonzero()[:, 0]
    
    return label(d, X, ind_class0, ind_class1, N, V, binary)

# --------------------------------------------

# ============================================
# Create labels
# ============================================
 
def label(d, X, ind_class0, ind_class1, N, V, binary):
    """Auxiliary function to create labels for toydata.
    
    Parameters
    ----------
    d : int
        Dimension of data space. Returns d = 2.
    X : torch.Tensor
        All data. Shape (N+V, 2).
    ind_class0 : torch.Tensor
        Indices of data in X that belongs to class 0. Shape 
        (num_samples_of_class0).
    ind_class1 : torch.Tensor
        Indices of data in X that belongs to class 1. Shape 
        (num_samples_of_class1).
    N : int
        Number of training samples.
    V : int
        Number of validation samples.
    binary : bool
        Create labels as for binary classification, otherwise as for general 
        classification.
        
    Returns
    -------
    X_train : torch.Tensor
        Training data. Shape (N, 2). 
    C_train : torch.Tensor
        Lables for training data. Shape (N, K).
    X_val : torch.Tensor
        Validation data. Shape (V, 2).
    C_val : torch.Tensor
        Lables for validation data. Shape (V, K).
    d : int
        Dimension of data space. Returns d = 2.
    K : int
        Number of classes. Returns K = 1 if treated as binary classification, 
        and K = 2 if treated as general classification.
    """
    if binary == True:
        K = 1
        C = torch.zeros(N + V, K)
        C[ind_class0, :] = 0.0
        C[ind_class1, :] = 1.0
    else:
        K = 2
        C = torch.zeros(N + V, K)
        C[ind_class0, :] = torch.tensor([1.0, 0.0])
        C[ind_class1, :] = torch.tensor([0.0, 1.0])

    X_train = X[:N, :]
    X_val = X[N:, :]
    C_train = C[:N, :]
    C_val = C[N:, :]

    return [X_train, C_train, X_val, C_val, d, K]