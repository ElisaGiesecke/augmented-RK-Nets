# ============================================
# Compute accuracy
# ============================================

def acc(C,C_pred):
    """Compute the accuracy by comparing the prediction against the ground 
    truth. 
    
    Parameters
    ----------
    C : torch.Tensor
        True lables of data samples. Shape (num_samples, K).
    C_pred : torch.Tensor
        Predicted lables of data samples. Shape (num_samples, K).
        
    Returns
    -------
    float
        Accuracy as percentage.
    """
    # check that true and predicted labels are of same size
    assert C.size() == C_pred.size()
    
    total,K = C.size()
    correct = (C_pred == C).sum()/K
    
    return 100 * correct / total