import numpy as np
import torch

# ============================================
# Multiclass donut toydata
# ============================================

def donut_multiclass(N=1500, V=1500, d=2, K=3):
    """Generate toydataset with multiple classes of high dimensional donut.
    
    Parameters
    ----------
    N : int, optional
        Number of training samples. Default is N = 1500.
    V : int, optional
        Number of validation samples. Default is V = 1500.
    d : int
        Dimension of data space. Default is d = 2.
    K : int
        Number of classes. Default is K = 3.
        
    Returns
    -------
    X_train : torch.Tensor
        Training data. Shape (N, d). 
    C_train : torch.Tensor
        Lables for training data. Shape (N, K).
    X_val : torch.Tensor
        Validation data. Shape (V, d).
    C_val : torch.Tensor
        Lables for validation data. Shape (V, K).
    d : int
        Dimension of data space.
    K : int
        Number of classes. 
    
    Notes
    -----
    Labels are always given in general form because binary labels work only for
    2 classes.
    Number of classes K has to be chosen greater than or equal to 2. Trivial 
    case of only one class should be avoided since K = 1 is interpreted as 2 
    classes with binary labels. So functions acting on single class dataset 
    will not work.
    """
    X = torch.randn(N + V, d)*1.5
    R = torch.sqrt(torch.sum(X ** 2, dim=1))
    tresholds = np.linspace(0.0, 4.0, num=K, endpoint=False)
    ind_class_all = []
    for i in range(K):
        if i < K-1:
            ind_class = ((R >= tresholds[i]) * (R < tresholds[i + 1])).nonzero()[:, 0]
        else:
            ind_class = (R >= tresholds[i]).nonzero()[:, 0]
        ind_class_all.append(ind_class)
    X.add_(0.2 * torch.randn_like(X))        # add some noise to the data
    
    return label(d, X, ind_class_all, N, V, K)

# ============================================
# Multiclass squares toydata 
# ============================================

def squares_multiclass(N=1500, V=1500, d=2): 
    """Generate toydataset of squares with 4 classes in 2D or 3D.
    
    Parameters
    ----------
    N : int, optional
        Number of training samples. Default is N = 1500.
    V : int, optional
        Number of validation samples. Default is V = 1500.
    d : int
        Dimension of data space, either 2 or 3. Default is d = 2.
        
    Returns
    -------
    X_train : torch.Tensor
        Training data. Shape (N, d). 
    C_train : torch.Tensor
        Lables for training data. Shape (N, K).
    X_val : torch.Tensor
        Validation data. Shape (V, d).
    C_val : torch.Tensor
        Lables for validation data. Shape (V, K).
    d : int
        Dimension of data space.
    K : int
        Number of classes, here K = 4.
    """
    K = 4
    X = 8 * torch.rand(N + V, d) - 4
        
    ind_class_all = []
    if d == 2:
        ind_class = ((X[:, 0] > 0) * (X[:, 1] > 0)).nonzero()[:, 0]
        ind_class_all.append(ind_class)
        ind_class = ((X[:, 0] > 0) * (X[:, 1] <= 0)).nonzero()[:, 0]
        ind_class_all.append(ind_class)
        ind_class = ((X[:, 0] <= 0) * (X[:, 1] > 0)).nonzero()[:, 0]
        ind_class_all.append(ind_class)
        ind_class = ((X[:, 0] <= 0) * (X[:, 1] <= 0)).nonzero()[:, 0]
        ind_class_all.append(ind_class)
    elif d == 3:
        ind_class = ((X[:, 0] > 0) * (X[:, 1] > 0) * (X[:, 2] > 0) + (X[:, 0] <= 0) * (X[:, 1] <= 0) * (X[:, 2] <= 0)).nonzero()[:, 0]
        ind_class_all.append(ind_class)
        ind_class = ((X[:, 0] > 0) * (X[:, 1] > 0) * (X[:, 2] <= 0) + (X[:, 0] <= 0) * (X[:, 1] <= 0) * (X[:, 2] > 0)).nonzero()[:, 0]
        ind_class_all.append(ind_class)
        ind_class = ((X[:, 0] > 0) * (X[:, 1] <= 0) * (X[:, 2] <= 0) + (X[:, 0] <= 0) * (X[:, 1] > 0) * (X[:, 2] > 0)).nonzero()[:, 0]
        ind_class_all.append(ind_class)
        ind_class = ((X[:, 0] > 0) * (X[:, 1] <= 0) * (X[:, 2] > 0) + (X[:, 0] <= 0) * (X[:, 1] > 0) * (X[:, 2] <= 0)).nonzero()[:, 0]
        ind_class_all.append(ind_class)
    else:
        raise Exception('dimension of data space has to be either 2 or 3')
        
    X.add_(0.3 * torch.randn_like(X))        # add some noise to the data
    
    return label(d, X, ind_class_all, N, V, K)


# --------------------------------------------

# ============================================
# Create labels
# ============================================
 
def label(d, X, ind_class_all, N, V, K):
    """Auxiliary function to create labels for toydata.
    
    Parameters
    ----------
    d : int
        Dimension of data space.
    X : torch.Tensor
        All data. Shape (N+V, d).
    ind_class_all : list of torch.Tensor
        The ith entry of list are indices of data in X that belongs to class i. 
        Shape of entries (num_samples_of_class_i).
    N : int
        Number of training samples.
    V : int
        Number of validation samples.
    K : int
        Number of classes.
        
    Returns
    -------
    X_train : torch.Tensor
        Training data. Shape (N, d). 
    C_train : torch.Tensor
        Lables for training data. Shape (N, K).
    X_val : torch.Tensor
        Validation data. Shape (V, d).
    C_val : torch.Tensor
        Lables for validation data. Shape (V, K).
    d : int
        Dimension of data space.
    K : int
        Number of classes.
    """
    C = torch.zeros(N + V, K)
    for i, ind_class in enumerate(ind_class_all):
        C[ind_class, i] = 1.0

    X_train = X[:N, :]
    X_val = X[N:, :]
    C_train = C[:N, :]
    C_val = C[N:, :]

    return [X_train, C_train, X_val, C_val, d, K]