import torch
import torch.optim as optim
from statistics import mean 
import loss
import prediction
import accuracy

# ============================================
# Train a network and generate statistics (cost and accuracy)
# ============================================

def train(net, TrainLoader, all_data, optim_method='SGD',lr=1e-1,
          weight_decay=0.0, momentum=0.0, loss_criterion='cross_entropy',
          max_epochs=40, tol=5e-2, print_batch=50):
    """Train a network and generate statistics (cost and accuracy). 
    
    Parameters
    ----------
    net : torch.nn.Module
        Neural network to train.
    TrainLoader :  torch.utils.data.DataLoader
        Dataloader for training. Used for iterating over training data samples 
        and their labels batch-wise.
    data : list of torch.Tensor
        Contains the training and validation data XTrain and XVal and their 
        labels CTrain and CVal, respectively. data = [XTrain,CTrain,XVal,CVal].
    optim_method : str, optional
        Specifies the method for optimization. Either 'sgd', 'adagrad', 'adam',
        or 'adadelta'. Default is optimizerMethod = 'sgd'.
    lr : float, optional
        Learning rate. Only relevant for 'sgd'. Default is lr = 1e-1.
    weight_decay : float, optional
        Weight decay for regularization. Default is weight_decay = 0.0.
    momentum : float, optional
        Momentum. Only relevant for 'sgd'. Default is momentum = 0.0.
    loss_criterion : str, optional
        Loss function to use. Either 'mse' or 'cross_entropy'. Default is 
        lossCriterion = 'cross_entropy'.
    max_epochs: int, optional
        Maximum number of epochs for training. Default is maxEpochs = 40.
    tol : float, optional
        Minimum decrease of performance (measured by validation cost) for early
        stopping to avoid overfitting. Default is tol = 5e-2.
    print_batch : int, optional
        Number of batches after which training loss and gradient is printed 
        frequently during training. If print_batch = len(TrainLoader), prints
        only once at the end of each epoch. Default is print_batch = 50.
    
    Returns
    -------
    cost_train : list of torch.Tensor
        Training costs after each epoch. Entries of Shape (1).
    acc_train : list of torch.Tensor
        Training accuracy after each epoch. Entries of Shape (1).
    cost_val : list of torch.Tensor
        Validation costs after each epoch. Entries of Shape (1).
    acc_val : list of torch.Tensor
        Validation accuracy after each epoch. Entries of Shape (1).
    
    Notes
    -----
    Function prints at specified frequency:
    [epoch, i_batch] training loss: running_loss    gradient: running_gradient 
    """
    X_train, C_train, X_val, C_val = all_data
    cost_train = []
    acc_train = []
    cost_val = []
    acc_val = []
    
    # store statistics before training
    with torch.no_grad():                                # prevent tracking history 
        X_predicted, X_classified, _ = net(X_train)      # propagate training data through network to obtain output
        C_pred = prediction.pred(X_predicted)            # get prediction from network output
        cost_train.append(loss.cost_value(X_predicted, X_classified, C_train,
                                        loss_criterion).item())
        acc_train.append(accuracy.acc(C_train, C_pred).item())
    
        X_predicted, X_classified, _ = net(X_val)         # repeat for validation data
        C_pred = prediction.pred(X_predicted)
        cost_val.append(loss.cost_value(X_predicted, X_classified, C_val,
                                      loss_criterion).item())
        acc_val.append(accuracy.acc(C_val, C_pred).item())
    
    # get optimizer
    if optim_method == 'sgd':
        optimizer = optim.SGD(net.parameters(), lr=lr, 
                              weight_decay=weight_decay, momentum=momentum)
    elif optim_method == 'adagrad':
        optimizer = optim.Adagrad(net.parameters(), weight_decay=weight_decay)
    elif optim_method == 'adam':
        optimizer = optim.Adam(net.parameters(), weight_decay=weight_decay)
    elif optim_method == 'adadelta':
        optimizer = optim.Adadelta(net.parameters(), weight_decay=weight_decay)
    else:
        raise Exception('invalid optimization method')
    
    for epoch in range(max_epochs):      # loop over the dataset multiple times
        running_loss = 0.0
        running_gradient = 0.0
    
        for i_batch, sample_batched in enumerate(TrainLoader):
            inputs, labels = sample_batched['inputs'], sample_batched['labels']
            optimizer.zero_grad()                                # set parameter gradients to zero
            outputs_predicted, outputs_classified, _ = net(inputs)   # forward pass
            cost_batched = loss.cost_value(outputs_predicted, 
                                           outputs_classified, labels, 
                                           loss_criterion)       # compute value of cost function
            cost_batched.backward()                              # backward pass
            optimizer.step()                                     # update parameters

            # update running loss and gradient
            running_loss += cost_batched.item()
            running_gradient += sum(torch.norm(p.grad) for p in 
                                    net.parameters() if p.requires_grad)
            
            # print running loss and gradient every print_batch mini-batches
            if i_batch % print_batch == print_batch - 1:
                print('[%d, %5d] training loss: %.3f    gradient: %.3f' 
                      % (epoch + 1, i_batch + 1, running_loss/print_batch, 
                         running_gradient/print_batch))
                running_loss = 0.0 
                running_gradient = 0.0
                
        # store statistics at the end of epoch
        with torch.no_grad():
            X_predicted, X_classified, _ = net(X_train)               
            C_pred = prediction.pred(X_predicted)
            cost_train.append(loss.cost_value(X_predicted, X_classified, 
                                              C_train, loss_criterion).item())
            acc_train.append(accuracy.acc(C_train, C_pred).item())
        
            X_predicted, X_classified, _ = net(X_val)                       
            C_pred = prediction.pred(X_predicted)
            cost_val.append(loss.cost_value(X_predicted, X_classified, C_val,
                                          loss_criterion).item())
            acc_val.append(accuracy.acc(C_val, C_pred).item())
        
        # early stopping if overfitting occurs (validation cost as measure of performance)
        if epoch >= 10 and mean(cost_val[-5:]) - mean(cost_val[-10:-5]) > tol:  # comparison of average over 5 epochs
            break
        
    print('Finished Training')
    return [cost_train, acc_train, cost_val, acc_val]
