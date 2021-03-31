import torch

# ============================================
# Get prediction from network output
# ============================================

def pred(X_predicted):
    """Get the prediction from the network output.  
    
    Parameters
    ----------
    X_predicted : torch.Tensor
        Probability for each class of data samples (final network output 
        after applying classifier and hypothesis function). Shape (num_samples,
        K).
        
    Returns
    -------
    torch.Tensor
        Predicted labels of data samples. Shape (num_samples, K).
    """
    # binary label
    if X_predicted.size()[1] == 1: 
        ind_class1 = (X_predicted >= 0.5).nonzero()[:, 0]
        C_pred = torch.zeros_like(X_predicted)
        C_pred[ind_class1, :] = 1.0
        
    # general label
    else:         
        ind_prediction = torch.argmax(X_predicted, dim=1)
        C_pred = torch.zeros_like(X_predicted)
        C_pred[torch.arange(X_predicted.size()[0]), ind_prediction] = 1.0

    return C_pred