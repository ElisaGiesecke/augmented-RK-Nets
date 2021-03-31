import torch.nn as nn

# ============================================
# Compute value of cost function 
# ============================================

def cost_value(X_predicted, X_classified, C, loss_criterion):  
    """Compute the value of the cost function.  
    
    Parameters
    ----------
    X_predicted : torch.Tensor
        Probability for each class of data samples (final network output 
        after applying classifier and hypothesis function). Shape (num_samples,
        K).
    X_classified : torch.Tensor
        Intermediate network output after applying classifier but before 
        applying hypothesis function. 
        Shape (num_samples, K).
    C : torch.Tensor
        Lables of data samples. Shape (num_samples, K).
    loss_criterion : str
        Loss function to use. Either 'mse' or 'cross_entropy'.
        
    Returns
    -------
    torch.Tensor
        Value of cost function. Shape (1).
    """
    # choice of loss function
    if loss_criterion == 'mse':
        cost = mse(X_predicted, C)
        
    elif loss_criterion == 'cross_entropy':
        # binary label
        if C.size()[1] == 1: 
            cost = cross_entropy_binary(X_predicted, C)
        # general label
        else:
            cost = cross_entropy_general(X_classified, C)
            
    else:
        raise Exception('invalid loss function')
        
    return cost
    
# --------------------------------------------
    
# ============================================
# Mean squared error loss 
# (on prediction and binary or genereal label)
# ============================================

def mse(X_predicted, C):
    """Compute the value of the mean squared error loss for binary or general 
    label.
    
    Parameters
    ----------
    X_predicted : torch.Tensor
        Probability for each class of data samples (final network output 
        after applying classifier and hypothesis function). Shape (num_samples,
        K).
    C : torch.Tensor
        Lables of data samples. Shape (num_samples, K).
        
    Returns
    -------
    torch.Tensor
        Value of mean squared error loss. Shape (1).
    """
    loss_func = nn.MSELoss()
       
    # compute value of loss function
    return loss_func(X_predicted, C)
    
    
# ============================================
# Cross entropy loss for binary label
# (on prediction and binary label)
# ============================================

def cross_entropy_binary(X_predicted, C):
    """Compute the value of the cross entropy loss for binary label.  
    
    Parameters
    ----------
    X_predicted : torch.Tensor
        Probability for each class of data samples (final network output 
        after applying classifier and hypothesis function). Shape (num_samples,
        K).
    C : torch.Tensor
        Lables of data samples. Shape (num_samples, K).
        
    Returns
    -------
    torch.Tensor
        Value of cross entropy loss. Shape (1).
    """
    loss_func = nn.BCELoss()
    
    # compute value of loss function
    return loss_func(X_predicted, C)
    
    
# ============================================
# Cross entropy loss for general label
# (on classified data (unnormalized) and number of target class)
# ============================================

def cross_entropy_general(X_classified, C):
    """Compute the value of the cross entropy loss for general label.  
    
    Parameters
    ----------
    X_classified : torch.Tensor
        Intermediate network output after applying classifier but before 
        applying hypothesis function. 
        Shape (num_samples, K).
    C : torch.Tensor
        Lables of data samples. Shape (num_samples, K).
        
    Returns
    -------
    torch.Tensor
        Value of cross entropy loss. Shape (1).
    """
    loss_func = nn.CrossEntropyLoss()
    
    # create target out of label C
    target = C.nonzero()[:, 1]
    
    # compute value of loss function
    return loss_func(X_classified, target)
    
 