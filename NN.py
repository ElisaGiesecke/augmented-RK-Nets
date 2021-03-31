import torch
import torch.nn as nn

# ============================================
# Fully-connected feed-forward network 
# ============================================

class StandardNet(nn.Module):
    """Fully-connected feed-forward network.
    
    Parameters
    ----------
    d : int
        Dimension of data space.
    d_hat : int
        Dimension of feature space (here constant over all layers).
    K : int
        Number of classes. K = 1 if treated as binary classification, 
        and K >= 2 if treated as general classification. 
    L : int
        Number of layers.
    act_function : str
        Activation function. Either 'tanh', 'sigmoid', 'relu' or 'softplus'.
        
    Notes
    -----
    Input layer corresponds to space augmentation if necessary. For that 
    reason, it has no bias and training is usually disabled manually outside of
    this class. Instead of using random initialization, the weight can be set
    manually outside of this class, e. g. to augment the space by padding with 
    zeros.
    Classifier is an affine function with randomly initialized weight and bias.
    As for the input layer, training can be disabled and weight and/or bias set 
    manually outside of this class.
    Hypothesis function is given by sigmoid function in case of binary labels 
    and given by softmax function in case of general labels.
    """
    def __init__(self, d, d_hat, K, L, act_function):
        super(StandardNet, self).__init__()
        
        self.d_hat = d_hat
        self.L = L
        self.act_function = act_function
        
        # L layers defined by affine operation z = Ky + b
        layers = []
        
        # input layer with no bias 
        layers.append(nn.Linear(d, d_hat, bias=False))        
        
        for l in range(L):
            layers.append(nn.Linear(d_hat, d_hat))
        self.layers = nn.ModuleList(layers)
        
        # affine classifier
        self.classifier = nn.Linear(d_hat, K)
        
        # activation function
        activations = nn.ModuleDict([['tanh', nn.Tanh()],
                                      ['sigmoid', nn.Sigmoid()], 
                                      ['relu', nn.ReLU()],
                                      ['softplus', nn.Softplus()]])
        try:
            self.act = activations[act_function]
        except KeyError as e:
            raise type(e)('invalid activation function')
        
        # hypothesis function to normalize prediction
        if K == 1:
            self.hyp = nn.Sigmoid()
        else:
            self.hyp = nn.Softmax(dim=1)
        
    def forward(self, X):
        """Propagate data through network. 
        
        Parameters
        ----------
        X : torch.Tensor
            Data samples propagated through network. Shape (num_samples, d).
        
        Returns
        -------
        X_predicted : torch.Tensor
            Probability for each class of data samples (final network output 
            after applying classifier and hypothesis function). Shape 
            (num_samples, K).
        X_classified : torch.Tensor
            Intermediate network output after applying classifier but before 
            applying hypothesis function. Shape (num_samples, K).
        X_transformed : torch.Tensor
            Features corresponding to data samples at each layer. Shape 
            (num_samples, d_hat, L+1).
        """
        # track transformation of features
        X_transformed = torch.empty(X.shape[0], self.d_hat, self.L + 1)
        
        # propagate data through network (forward pass)
        for i, layer in enumerate(self.layers):
            # input layer with no activation 
            if i == 0:
                X = layer(X)
            else:
                X = self.act(layer(X))
            X_transformed[:, :, i] = X
        
        # apply classifier
        X_classified = self.classifier(X)
        
        # apply hypothesis function
        X_predicted = self.hyp(X_classified)
        
        return [X_predicted, X_classified, X_transformed]
    
# ============================================
# Euler network
# Residual neural network, if h=1 
# ============================================
     
class EulerNet(nn.Module):
    """Euler network/Residual neural network. 
    
    Parameters
    ----------
    d : int
        Dimension of data space.
    d_hat : int
        Dimension of feature space (here constant over all layers).
    K : int
        Number of classes. K = 1 if treated as binary classification, 
        and K >= 2 if treated as general classification. 
    L : int
        Number of layers.
    act_function : str
        Activation function. Either 'tanh', 'sigmoid', 'relu' or 'softplus'.
        
    Notes
    -----
    Input layer corresponds to space augmentation if necessary. For that 
    reason, it has no bias and training is usually disabled manually outside of
    this class. Instead of using random initialization, the weight can be set
    manually outside of this class, e. g. to augment the space by padding with 
    zeros.
    Classifier is an affine function with randomly initialized weight and bias.
    As for the input layer, training can be disabled and weight and/or bias set 
    manually outside of this class.
    Hypothesis function is given by sigmoid function in case of binary labels 
    and given by softmax function in case of general labels.
    """
    def __init__(self, d, d_hat, K, L, act_function):
        super(EulerNet, self).__init__()
        
        self.d_hat = d_hat
        self.L = L
        self.act_function = act_function
        
        # L layers defined by affine operation z = Ky + b
        layers = []
        
        # input layer with no bias 
        layers.append(nn.Linear(d, d_hat, bias=False))        
        
        for l in range(L):
            layers.append(nn.Linear(d_hat, d_hat))
        self.layers = nn.ModuleList(layers)
        
        # affine classifier
        self.classifier = nn.Linear(d_hat, K)
        
        # activation function
        activations = nn.ModuleDict([['tanh', nn.Tanh()],
                                      ['sigmoid', nn.Sigmoid()],
                                      ['relu', nn.ReLU()],
                                      ['softplus', nn.Softplus()]])
        self.act = activations[act_function]
        
        # hypothesis function to normalize prediction
        if K == 1:
            self.hyp = nn.Sigmoid()
        else:
            self.hyp = nn.Softmax(dim=1)
        
    def forward(self, X):
        """Propagate data through network. 
        
        Parameters
        ----------
        X : torch.Tensor
            Data samples propagated through network. Shape (num_samples, d).
        
        Returns
        -------
        X_predicted : torch.Tensor
            Probability for each class of data samples (final network output 
            after applying classifier and hypothesis function). Shape 
            (num_samples, K).
        X_classified : torch.Tensor
            Intermediate network output after applying classifier but before 
            applying hypothesis function. Shape (num_samples, K).
        X_transformed : torch.Tensor
            Features corresponding to data samples at each layer. Shape 
            (num_samples, d_hat, L+1).
        """
        # choose step size
        h = 0.05
        
        # track transformation of features
        X_transformed = torch.empty(X.shape[0], self.d_hat, self.L + 1)
        
        # propagate data through network (forward pass)
        for i, layer in enumerate(self.layers):
            # input layer with no activation 
            if i == 0:
                X = layer(X)
            # residual layers
            else:
                X = X + h * self.act(layer(X))     
            X_transformed[:, :, i] = X
        
        # apply classifier
        X_classified = self.classifier(X)
        
        # apply hypothesis function
        X_predicted = self.hyp(X_classified)
        
        return [X_predicted, X_classified, X_transformed]

# ============================================
# Runge-Kutta 4 network 
# ============================================

class RK4Net(nn.Module):
    """Runge-Kutta 4 network. 
    
    Parameters
    ----------
    d : int
        Dimension of data space.
    d_hat : int
        Dimension of feature space (here constant over all layers).
    K : int
        Number of classes. K = 1 if treated as binary classification, 
        and K >= 2 if treated as general classification. 
    L : int
        Number of layers.
    act_function : str
        Activation function. Either 'tanh', 'sigmoid', 'relu' or 'softplus'.
        
    Notes
    -----
    Input layer corresponds to space augmentation if necessary. For that 
    reason, it has no bias and training is usually disabled manually outside of
    this class. Instead of using random initialization, the weight can be set
    manually outside of this class, e. g. to augment the space by padding with 
    zeros.
    Classifier is an affine function with randomly initialized weight and bias.
    As for the input layer, training can be disabled and weight and/or bias set 
    manually outside of this class.
    Hypothesis function is given by sigmoid function in case of binary labels 
    and given by softmax function in case of general labels.
    """
    def __init__(self, d, d_hat, K, L, act_function):
        super(RK4Net, self).__init__()
        
        self.d_hat = d_hat
        self.L = L
        self.act_function = act_function
        
        # L layers defined by affine operation z = Ky + b
        layers = []
        
        # input layer with no bias 
        layers.append(nn.Linear(d, d_hat, bias=False))        
        
        for l in range(L):
            layers.append(nn.Linear(d_hat, d_hat))
        self.layers = nn.ModuleList(layers)
        
        # affine classifier
        self.classifier = nn.Linear(d_hat, K)
        
        # activation function
        activations = nn.ModuleDict([['tanh', nn.Tanh()],
                                      ['sigmoid', nn.Sigmoid()],
                                      ['relu', nn.ReLU()],
                                      ['softplus', nn.Softplus()]])
        self.act = activations[act_function]
        
        # hypothesis function to normalize prediction
        if K == 1:
            self.hyp = nn.Sigmoid()
        else:
            self.hyp = nn.Softmax(dim=1)
        
    def forward(self, X):
        """Propagate data through network. 
        
        Parameters
        ----------
        X : torch.Tensor
            Data samples propagated through network. Shape (num_samples, d).
        
        Returns
        -------
        X_predicted : torch.Tensor
            Probability for each class of data samples (final network output 
            after applying classifier and hypothesis function). Shape 
            (num_samples, K).
        X_classified : torch.Tensor
            Intermediate network output after applying classifier but before 
            applying hypothesis function. Shape (num_samples, K).
        X_transformed : torch.Tensor
            Features corresponding to data samples at each layer. Shape 
            (num_samples, d_hat, L+1).
        """
        # choose step size
        h = 0.05
        
        # track transformation of features
        X_transformed = torch.empty(X.shape[0], self.d_hat, self.L + 1)
        
        # propagate data through network (forward pass)
        for i, layer in enumerate(self.layers):
            # input layer with no activation 
            if i == 0:
                X = layer(X)
            # RK4 layers
            else:
                f1 = self.act(layer(X))
                f2 = self.act(layer(X + h / 2.0 * f1))  
                f3 = self.act(layer(X + h / 2.0 * f2))  
                f4 = self.act(layer(X + h * f3)) 
                X = X + h * (f1 / 6.0 + f2 / 3.0 + f3 / 3.0 + f4 / 6.0) 
            X_transformed[:, :, i] = X
        
        # apply classifier
        X_classified = self.classifier(X)
        
        # apply hypothesis function
        X_predicted = self.hyp(X_classified)
        
        return [X_predicted, X_classified, X_transformed]