import numpy as np
import matplotlib.pyplot as plt        
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import colors
import torch
from sklearn.decomposition import PCA
import imageio
import os

# ============================================
# Plot toydata
# ============================================

def plot_multiclass_toydata(X, C):
    """Plot points of toydata with multiple classes (max. 6) as scatter plot. 
    Classes are represented by colors in the following order: blue, red, 
    yellow, green, magenta and cyan.
    
    Parameters
    ----------
    X : torch.Tensor
        Data samples. Shape (num_samples, d).
    C : torch.Tensor
        Lables of data samples. Shape (num_samples, K).
    """ 
    # transform binary in general label if necessary
    if C.size()[1] == 1:
        C = transform_label(C)
    K = C.size()[1]
    
    # colors representing classes    
    class_colors = ['.b', '.r', '.y', '.g', '.m', '.c']
    assert K <= len(class_colors),'maximum number of classes is '+ str(len(class_colors))
    
    # get indices for multiple classes with general label
    ind_class_all = []
    for i in range(K):
        ind_class = C[:, i].nonzero()[:, 0]
        ind_class_all.append(ind_class)
       
    # create scatter plot in 2D or 3D
    if X.size()[1] == 2:  
        for i, ind_class in enumerate(ind_class_all):
            plt.plot(X.numpy()[ind_class, 0], X.numpy()[ind_class, 1], 
                     class_colors[i])
        plt.axis((-4, 4, -4, 4))
        plt.gca().set_aspect('equal', adjustable='box')  
        plt.xticks(np.arange(-4, 4.1, step=2))
        plt.yticks(np.arange(-4, 4.1, step=2))
    elif X.size()[1] == 3:
        ax = plt.axes(projection='3d')
        for i, ind_class in enumerate(ind_class_all):
            ax.plot(X.numpy()[ind_class, 0], X.numpy()[ind_class, 1], 
                    X.numpy()[ind_class, 2], class_colors[i])
        ax.set_xlim(-4, 4), ax.set_ylim(-4, 4), ax.set_zlim(-4, 4)  
        ax.set_xticks(np.arange(-4, 4.1, step=2))
        ax.set_yticks(np.arange(-4, 4.1, step=2))
        ax.set_zticks(np.arange(-4, 4.1, step=2))

# ============================================
# Plot coloured background according to prediction
# ============================================

def plot_multiclass_prediction(net, x_min=-4, x_max=4, y_min=-4, y_max=4):
    """Plot coloured background for multiple classes according to prediction of
    network as contour plot. Classes are represented by colors in the following
    order: blue, red, yellow, green, magenta and cyan. Works only for d = 2,
    i.e. 2D data.
    
    Parameters
    ----------
    net : torch.nn.Module
        Neural network used for prediction.
    x_min, x_max, y_min, y_max : int or float, optional
        Domain in which to plot coloured background. Default is 
        x_min = y_min = -4 and x_max = y_max = 4.
        
    Notes
    -----
    Scatter plot produced by the function plot_multiclass_toydata and contour 
    plot produced by this function can be combined in a single plot.
    """
    # create grid
    x_grid, y_grid = np.meshgrid(np.linspace(x_min, x_max, 80), 
                                 np.linspace(y_min, y_max, 80))
    XGrid = torch.tensor(list(zip(x_grid.flatten(), y_grid.flatten()))).float()
    X_predicted, _, _ = net(XGrid)
    X_predicted = X_predicted.detach()
    
    # transform binary in general label if necessary
    if X_predicted.size()[1] == 1:
        X_predicted = transform_label(X_predicted)
    K = X_predicted.size()[1]
    
    # colors representing classes
    a = 0.2     # tune brightness and darkness of non-classified region (try also 0.0)
    b = 0.8     # tune intensity of classified region (try also 1.)
    class_colors = [[(a, a, b, 0.), (a, a, b, 1.)],
                     [(b, a, a, 0.), (b, a, a, 1.)],
                     [(b, b, a, 0.), (b, b, a, 1.)],
                     [(a, b, a, 0.), (a, b, a, 1.)],
                     [(b, a, b, 0.), (b, a, b, 1.)],
                     [(a, b, b, 0.), (a, b, b, 1.)]]
    levels = np.linspace(0.0, 1.0, 100)
    assert K <= len(class_colors), 'maximum number of classes is ' + str(len(class_colors))
        
    # create contour plot
    for i in range(K):
        prediction_grid = X_predicted[:, i].reshape(80, 80)
        transcolor = colors.LinearSegmentedColormap.from_list(
            name='Transcolor', colors=class_colors[i])
        cnt = plt.contourf(x_grid, y_grid, prediction_grid, levels, cmap=transcolor)#, alpha=0.5)
        for c in cnt.collections:        # hide contour lines
            c.set_edgecolor("face")
            c.set_linewidth(0.000000000001)            
    
# ============================================
# Plot transformation in feature space 
# (with coloured background according to prediction if 2D)
# with option to create video
# ============================================

def plot_multiclass_transformation(X_transformed, C, dim_plot, dim_reduction, 
                        constant_axes, output_layer=None, net=None, 
                        show_output=False, save=None):
    """Plot transformation in feature space for multiple classes. Reduce 
    dimensions either by Principal Component Analysis or by choosing features 
    if necessary, before plotting in 2D or 3D. If plotted in 2D, coloured 
    background according to prediction can be added either only in the output 
    layer or in all layers. Classes are represented by colors in the following
    order: blue, red, yellow, green, magenta, cyan, orange, gray, brown and 
    lawngreen. Hence, the function can be used for at most 10 classes and if a 
    coloured background is added, for at most 6 classes.
    
    Parameters
    ----------
    X_transformed : torch.Tensor
        Features corresponding to data samples at each layer. Shape 
        (num_samples, d_hat, L+1).
    C : torch.Tensor
        Lables of data samples. Shape (num_samples, K).
    dim_plot : int
        Dimension of plot, either 2 or 3.
    dim_reduction : bool
        Dimensionality reduction by principal component analysis if True.
        Otherwise choose features at random (here first 2 or 3).
    constant_axes : bool
        Keep axes constant over layers if True. Otherwise let axes vary.
    output_layer : bool or None
        If 2D plot, plot prediction background only in output layer if True,
        and plotted in all layers if False (requires constant axes).
    net : torch.nn.Module or None
        Neural network used for prediction (only required if 
        output_layer != None).
    show_output : bool
        Show only last plot of sequence, i.e. output layer. Default is False.
    save : str or None
        If None, shows sequence of plots. Otherwise, creates video and saves it
        as .gif file with name specified by string.
    
    Notes
    -----
    PCA is fitted new for each layer. Note that the projection produced by PCA
    is not unique, more precisely, signs can be flipped. This results in
    possible reflections of data across coordinate axes in subsequent layers.
    Plotting the prediction in the background makes only sense if 
    dimensionality is not reduced, i.e. if d_hat == dim_plot, so we need 
    d_hat = 2. We also get reasonable results if we use PCA for dimensionality
    reduction, i.e. if dim_reduction == True.
    Creates plt.fiure() and does plt.show().
    """
    assert dim_plot == 2 or dim_plot == 3, 'invalid dimension: transformation of features cannot be plotted'
    
    num_samples, d_hat, L_plus1 = X_transformed.size()
    assert d_hat >= dim_plot, 'dimension of feature space has to be larger than or equal to dimension of plot'
    
    X_transformed = X_transformed.detach().numpy()
    
    # transform binary in general label if necessary
    if C.size()[1] == 1:
        C = transform_label(C)
    K = C.size()[1]
    
    # colors representing classes    
    class_colors_scatter = ['b', 'r', 'y', 'g', 'm', 'c', 
                            'orange', 'gray','brown', 'lawngreen']
    a = 0.2     # tune brightness and darkness of non-classified region (try also 0.0)
    b = 0.8     # tune intensity of classified region (try also 1.)
    class_colors_contour = [[(a, a, b, 0.), (a, a, b, 1.)],
                     [(b, a, a, 0.), (b, a, a, 1.)],
                     [(b, b, a, 0.), (b, b, a, 1.)],
                     [(a, b, a, 0.), (a, b, a, 1.)],
                     [(b, a, b, 0.), (b, a, b, 1.)],
                     [(a, b, b, 0.), (a, b, b, 1.)]]
    levels = np.linspace(0.0, 1.0, 100)
    num_colors = np.minimum(len(class_colors_scatter), len(class_colors_contour))
    if isinstance(output_layer, bool):
        assert K <= num_colors, 'maximum number of classes is ' + str(num_colors)
    else:
        assert K <= len(class_colors_scatter), 'maximum number of classes is ' + str(len(class_colors_scatter))
    
    # get indices for multiple classes with general label
    ind_class_all = []
    for i in range(K):
        ind_class = C[:, i].nonzero()[:, 0]
        ind_class_all.append(ind_class)
    
    # reduce dimension via PCA or choosing features at random (here first 2 or 3 features)
    if dim_reduction:
        pca = PCA(n_components=dim_plot)     
        X_transformed_dim = np.empty([X_transformed.shape[0], dim_plot, L_plus1])
        for i in range(L_plus1 - 1):
            X_transformed_dim[:, :, i] = pca.fit_transform(X_transformed[:, :, i])
        pca.fit(X_transformed[:, :, L_plus1 - 1])
        X_transformed_dim[:, :, L_plus1 - 1] = pca.transform(X_transformed[:, :, L_plus1 - 1])            
    else:
        X_transformed_dim = np.empty([X_transformed.shape[0], dim_plot, L_plus1])
        for i in range(dim_plot):
            X_transformed_dim[:, i, :] = X_transformed[:, i, :]
    
    # determine domain
    if constant_axes:                      
        x_min = X_transformed_dim[:, 0, :].min()
        x_max = X_transformed_dim[:, 0, :].max()
        y_min = X_transformed_dim[:, 1, :].min()
        y_max = X_transformed_dim[:, 1, :].max()
        if dim_plot == 3:
            z_min = X_transformed_dim[:, 2, :].min()
            z_max = X_transformed_dim[:, 2, :].max()
    else:
        x_min = X_transformed_dim[:, 0, L_plus1 - 1].min()
        x_max = X_transformed_dim[:, 0, L_plus1 - 1].max()
        y_min = X_transformed_dim[:, 1, L_plus1 - 1].min()
        y_max = X_transformed_dim[:, 1, L_plus1 - 1].max()
        if dim_plot == 3:
            z_min = X_transformed_dim[:, 2, L_plus1 - 1].min()
            z_max = X_transformed_dim[:, 2, L_plus1 - 1].max()
    
    # for 2D plot
    if dim_plot == 2:   
        # create grid for prediction as coloured background
        x_grid, y_grid = np.meshgrid(np.linspace(x_min, x_max, 80), 
                                     np.linspace(y_min, y_max, 80))
        XGrid = torch.tensor(list(zip(x_grid.flatten(), y_grid.flatten()))).float()
        if dim_reduction:
            XGrid_original = pca.inverse_transform(XGrid)
            XGrid_original = torch.tensor(XGrid_original)
        else:
            XGrid_original = torch.zeros(XGrid.shape[0], d_hat)
            XGrid_original[:, [0, 1]] = XGrid    
      
        # apply linear classifier and hypothesis function to gridpoints in feature space
        prediction = net.hyp(net.classifier(XGrid_original)).detach()
        
        # transform binary in general label if necessary
        if prediction.size()[1] == 1:
            prediction = transform_label(prediction)            
        
        # plot features
        for i in range(L_plus1): 
            if show_output == False or i == L_plus1 - 1:
                Xi = X_transformed_dim[:, :, i]
                fig = plt.figure()
                for j in range(K):
                    plt.plot(Xi[ind_class_all[j], 0], Xi[ind_class_all[j], 1], 
                             color=class_colors_scatter[j], marker='.', linestyle='None')
                if constant_axes:
                    plt.axis((x_min, x_max, y_min, y_max))
                # plot prediction in background
                if output_layer == False or (output_layer == True and 
                                             i == L_plus1 - 1): 
                    for j in range(K):
                        transcolor = colors.LinearSegmentedColormap.from_list(
                                name='Transcolor', colors=class_colors_contour[j])
                        cnt = plt.contourf(x_grid, y_grid, prediction[:, j].reshape(80, 80), 
                                           levels, cmap=transcolor)#, alpha=0.5)
                        for c in cnt.collections:        # hide contour lines
                            c.set_edgecolor("face")
                            c.set_linewidth(0.000000000001)            
                        
                if save != None:
                    plt.savefig(save + '%d.png' % i, bbox_inches='tight')
                    plt.clf()
                    plt.close()
                else:
                    if i != L_plus1 - 1:
                        plt.show()
            
            
    # for 3D plot
    elif dim_plot == 3:
        # plot features
        for i in range(L_plus1): 
            if show_output == False or i == L_plus1 - 1:
                Xi = X_transformed_dim[:, :, i]
                ax = plt.axes(projection='3d')
                for j in range(K):
                    ax.plot(Xi[ind_class_all[j], 0], Xi[ind_class_all[j], 1], 
                            Xi[ind_class_all[j], 2], color=class_colors_scatter[j],
                            marker='.', linestyle='None')
                if constant_axes:
                    ax.set_xlim(x_min, x_max), ax.set_ylim(y_min, y_max), 
                    ax.set_zlim(z_min, z_max)                 
                        
                if save != None:
                    plt.savefig(save + '%d.png' % i, bbox_inches='tight')
                    plt.clf()
                    plt.close()
                else:
                    if i != L_plus1 - 1:
                        plt.show()
          
    else:
        raise Exception('invalid dimension: transformation of features cannot be plotted')
    
    # create video
    if save!= None:
        imgs = []
        for i in range(L_plus1):
            img_file = save + '%d.png' % i
            imgs.append(imageio.imread(img_file))
            os.remove(img_file)
        imageio.mimwrite(save + '.gif', imgs)
    
# ============================================
# Plot trajectories
# with option to create video
# ============================================

def plot_multiclass_trajectories(X_transformed, C, dim_plot, show_output=False, save=None):
    """Plot trajectories in feature space for multiple classes. Reduce 
    dimensions by choosing features if necessary (note that PCA does not make 
    sense, since it is fitted new for each layer), before plotting in 2D or 3D. 
    Axes are necessarily held constant over layers. Starting and ending points 
    of the trajectory are marked by points and circles, respectively. Classes 
    are represented by colors in the following order: blue, red, yellow, green, 
    magenta, cyan, orange, gray, brown and lawngreen. Hence, the function can 
    be used for at most 10 classes.
    
    Parameters
    ----------
    X_transformed : torch.Tensor
        Features corresponding to data samples at each layer. Shape 
        (num_samples, d_hat, L+1).
    C : torch.Tensor
        Lables of data samples. Shape (num_samples, K).
    dim_plot : int
        Dimension of plot, either 2 or 3.
    show_output : bool
        Show only last plot of sequence, i.e. complete trajectory. Default is 
        False.
    save : str or None
        If None, shows sequence of plots. Otherwise, creates video and saves it
        as .gif file with name specified by string.
        
    Notes
    -----
    For large datasets, it makes sense to choose only a few data samples in 
    order to make trajectories visible and not to overcrowd the plot.
    """
    assert dim_plot == 2 or dim_plot == 3, 'invalid dimension: transformation of features cannot be plotted'
    
    num_samples, d_hat, L_plus1 = X_transformed.size()
    assert d_hat >= dim_plot, 'dimension of feature space has to be larger than or equal to dimension of plot'
    
    X_transformed = X_transformed.detach().numpy()
    
    # transform binary in general label if necessary
    if C.size()[1] == 1:
        C = transform_label(C)
    K = C.size()[1]
    
    # colors representing classes    
    class_colors = ['b', 'r', 'y', 'g', 'm', 'c', 'orange', 'gray', 'brown', 'lawngreen']
    assert K <= len(class_colors), 'maximum number of classes is ' + str(len(class_colors))
    
    # get indices for multiple classes with general label
    ind_class_all = []
    for i in range(K):
        ind_class = C[:, i].nonzero()[:, 0]
        ind_class_all.append(ind_class)
    
    # reduce dimension choosing features at random (here first 2 or 3 features)
    X_transformed_dim = np.empty([num_samples, dim_plot, L_plus1])
    for i in range(dim_plot):
        X_transformed_dim[:, i, :] = X_transformed[:, i, :]
    
    # determine domain                      
    x_min = X_transformed_dim[:, 0, :].min()
    x_max = X_transformed_dim[:, 0, :].max()
    y_min = X_transformed_dim[:, 1, :].min()
    y_max = X_transformed_dim[:, 1, :].max()
    if dim_plot == 3:
        z_min = X_transformed_dim[:, 2, :].min()
        z_max = X_transformed_dim[:, 2, :].max()
    
    # for 2D plot
    if dim_plot == 2: 
        for i in range(L_plus1):
            if show_output == False or i == L_plus1 - 1:
                # plot starting (points) and ending (circles) points of trajectories
                for j in range(K):
                    plt.plot(X_transformed_dim[ind_class_all[j], 0, 0], 
                         X_transformed_dim[ind_class_all[j], 1, 0], 
                         color=class_colors[j], marker='.', linestyle='None')
                    plt.plot(X_transformed_dim[ind_class_all[j], 0, i], 
                         X_transformed_dim[ind_class_all[j], 1, i], 
                         color=class_colors[j], marker='o', linestyle='None')
                # plot trajectory for each point
                for k in range(num_samples):
                    for j in range(K):
                        if k in ind_class_all[j]:
                            plt.plot(X_transformed_dim[k, 0, :i + 1], 
                                 X_transformed_dim[k, 1, :i + 1], 
                                 color=class_colors[j], linewidth=0.5)
                # set axes
                plt.axis((x_min, x_max, y_min, y_max))
                
                if save != None:
                    plt.savefig(save + '%d.png' % i, bbox_inches='tight')
                    plt.clf()
                    plt.close()
                else:
                    if i != L_plus1 - 1:
                        plt.show()
            
    # for 3D plot
    elif dim_plot == 3:
        for i in range(L_plus1):
             if show_output == False or i == L_plus1 - 1:
                ax = plt.axes(projection='3d')
                # plot starting (points) and ending (circles) points of trajectories
                for j in range(K):
                    ax.plot(np.array(X_transformed_dim[ind_class_all[j], 0, 0],ndmin=1), 
                            np.array(X_transformed_dim[ind_class_all[j], 1, 0],ndmin=1), 
                            np.array(X_transformed_dim[ind_class_all[j], 2, 0],ndmin=1), 
                            color=class_colors[j], marker='.', linestyle='None')
                    ax.plot(np.array(X_transformed_dim[ind_class_all[j], 0, i],ndmin=1),
                            np.array(X_transformed_dim[ind_class_all[j], 1, i],ndmin=1),
                            np.array(X_transformed_dim[ind_class_all[j], 2, i],ndmin=1), 
                            color=class_colors[j], marker='o', linestyle='None')
                # plot trajectory for each point
                for k in range(num_samples):
                    for j in range(K):
                        if k in ind_class_all[j]:
                            ax.plot(X_transformed_dim[k, 0, :i + 1],
                                    X_transformed_dim[k, 1, :i + 1],
                                    X_transformed_dim[k, 2, :i + 1], 
                                    color=class_colors[j], linewidth=0.5)
                # set axes
                ax.set_xlim(x_min, x_max), ax.set_ylim(y_min, y_max), 
                ax.set_zlim(z_min, z_max)  
    
                if save != None:
                    plt.savefig(save + '%d.png' % i, bbox_inches='tight')
                    plt.clf()
                    plt.close()
                else:
                    if i != L_plus1 - 1:
                        plt.show()                 
    
    else:
        raise Exception('invalid dimension: trajectories cannot be plotted')
        
    # create video
    if save!= None:
        imgs = []
        for i in range(L_plus1):
            img_file = save + '%d.png' % i
            imgs.append(imageio.imread(img_file))
            os.remove(img_file)
        imageio.mimwrite(save + '.gif', imgs) 
        
# --------------------------------------------

# ============================================
# Transform binary label into general label
# ============================================

def transform_label(C):
    """Transform binary labels into general labels for toydata with 2 classes.
    
    Parameters
    ----------
    C : torch.Tensor
        Binary lables of data samples. Shape (num_samples, 1).
        
    Returns
    -------
    C_general : torch.Tensor
        General lables for data samples. Shape (N, 2).
    """
    C_general = torch.zeros(C.size(0),2)
    C_general[:,0] = 1.0-C[:,0]
    C_general[:,1] = C[:,0]
    return C_general