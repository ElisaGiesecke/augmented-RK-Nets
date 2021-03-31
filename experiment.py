import os
import time
import statistics 
import json
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import toydata
import toydata_highD
import toydata_multiclass
import NN
import dataset
import optimization
import prediction
import plot
import visualization

# ============================================
# Run experiment (with repetitions)
# ============================================

def run_experiment(data, network, training, reps=10, save_plots=True, 
                   save_results=True, save_statistics=True, 
                   main_directory=None):
    """Runs experiment with specified parameters several times. 
    Results are saved to files in experiment directory.
    
    Parameters
    ----------
    data : dict
        Parameters for data. Keys are 'data_example', 'N', 'V' and 'binary'.
    network : dict
        Parameters for network. Keys are 'net_architecture', 'd_hat', 'L', 
        'act_function', 'fill_zeros' and 'classifier_mode'.
    training : dict
        Paramters for training. Keys are 'loss_criterion', 'batch_size',
        'optim_method', 'lr', 'momentum', 'weight_decay', 'max_epochs' and 
        'tol'.
    reps : int
        Number of repetitions of experiment. Default is 10.
    save_plots : bool
        Save plots to folders data, prediction, transformation and 
        trajectories. Default is true.
    save_results : bool
        Save results to configuration.json, network_info.json, statistics.json
        and train_time.json. Default is true.
    save_statistics : bool
        Save statistics plots to .png files and statics tables to .txt files. 
        Default is true.
    main_directory : str or None
        If not None, creates main folder named as specified by string, in which
        to create folders for each experiment run. Default is None.
          
    Notes
    -----
    Example for values of parameter dictionaries can be found in 
    configs/simple_config.json
    """
    # collect experiment results in dictionary    
    results = {'configuration': {}, 'network_info': {},'statistics': {}, 
               'train_time': {}}
    
    results['configuration'] = {'data': data, 'network': network, 
           'training': training, 'reps': reps}
    results['network_info'] = {'num_neurons': [], 'num_params': [], 
           'num_trainable_params': []}
    results['statistics'] = {'cost_train': [], 'acc_train': [], 'cost_val': [], 
           'acc_val':[]}
    results['train_time'] = {'rep_time' : [], 'avg_time': None}
    
    if save_results or save_plots or save_statistics:
        timestamp = time.strftime('%d-%m-%Y_%H-%M')
        if main_directory != None:
            # create a folder (main directory) in which to create folders for each experiment run
            if not os.path.exists(main_directory):
                os.makedirs(main_directory)
            directory = main_directory + '/experiment_%s' % timestamp
        else:
            directory = 'experiment_%s' % timestamp
        # create a folder (experiment directory) to store experiment results 
        if not os.path.exists(directory):
            os.makedirs(directory)
               
    # run experiment reps times (train newly initialized network on newly 
    # generated dataset with same training procedure), store results and 
    # create plots if save_plots
    for i in range(reps):
        # generate toydata
        if data['data_example'] == 'donut_1D':
            X_train, C_train, X_val, C_val, d, K = toydata.donut_1D(data['N'], data['V'], data['binary'])
        elif data['data_example'] == 'donut_2D':
            X_train, C_train, X_val, C_val, d, K = toydata.donut_2D(data['N'], data['V'], data['binary'])
        elif data['data_example'] == 'squares':
            X_train, C_train, X_val, C_val, d, K = toydata.squares(data['N'], data['V'], data['binary'])
        elif data['data_example'] == 'spiral':
            X_train, C_train, X_val, C_val, d, K = toydata.spiral(data['N'], data['V'], data['binary'])
        elif data['data_example'] == 'spiral_3D':
            X_train, C_train, X_val, C_val, d, K = toydata_highD.spiral_3D(data['N'], data['V'], data['binary'])
        elif isinstance(data['data_example'], list): 
            if data['data_example'][0] == 'donut_highD':
                X_train, C_train, X_val, C_val, d, K = toydata_highD.donut_highD(data['N'], data['V'], data['binary'], data['data_example'][1])
            elif data['data_example'][0] == 'squares_highD':
                X_train, C_train, X_val, C_val, d, K = toydata_highD.squares_highD(data['N'], data['V'], data['binary'], data['data_example'][1])
            elif data['data_example'][0] == 'donut_multiclass':
                X_train, C_train, X_val, C_val, d, K = toydata_multiclass.donut_multiclass(data['N'], data['V'], data['data_example'][1], data['data_example'][2])
            elif data['data_example'][0] == 'squares_multiclass':
                X_train, C_train, X_val, C_val, d, K = toydata_multiclass.squares_multiclass(data['N'], data['V'], data['data_example'][1])   
        all_data = [X_train, C_train, X_val, C_val]
        
        # create neural network
        if network['net_architecture'] == 'StandardNet':
            net = NN.StandardNet(d, network['d_hat'], K, network['L'], network['act_function'])
        elif network['net_architecture'] == 'EulerNet':
            net = NN.EulerNet(d, network['d_hat'], K, network['L'], network['act_function'])
        elif network['net_architecture'] == 'RK4Net':
            net = NN.RK4Net(d, network['d_hat'], K, network['L'], network['act_function'])
        # specify space augmentation
        if network['fill_zeros']:
            trans_matrix = torch.zeros(network['d_hat'], d)
            trans_matrix[:d, :] = torch.eye(d)
            net.layers[0].weight = nn.Parameter(trans_matrix)
        net.layers[0].weight.requires_grad_(False)
        # train or fix classifier
        if network['classifier_mode'] == 'fix':
            net.classifier.weight.requires_grad_(False)          
            net.classifier.bias.requires_grad_(False)
        # get and store information about network (only once)
        if i == 0:
            num_neurons = d + network['d_hat'] * network['L']
            num_params = sum(p.numel() for p in net.parameters())
            num_trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
            results['network_info']['num_neurons'].append(num_neurons)
            results['network_info']['num_params'].append(num_params)
            results['network_info']['num_trainable_params'].append(num_trainable_params)
                                                                                
        # create dataset and dataloader
        TrainSet = dataset.toydataset(X_train, C_train)
        TrainLoader = DataLoader(TrainSet, training['batch_size'], shuffle=True) 
        print_batch = len(TrainLoader) 
                                            
        # train network and measure time
        start = time.time()
        cost_train, acc_train, cost_val, acc_val = optimization.train(net, TrainLoader, all_data, 
                                                                      training['optim_method'], training['lr'],
                                                                      training['weight_decay'], training['momentum'],
                                                                      training['loss_criterion'], training['max_epochs'],
                                                                      training['tol'], print_batch)
        end = time.time() 
        
        # store statistics and training time
        results['statistics']['cost_train'].append(cost_train)
        results['statistics']['acc_train'].append(acc_train)
        results['statistics']['cost_val'].append(cost_val)
        results['statistics']['acc_val'].append(acc_val)
        
        train_time = end - start
        results['train_time']['rep_time'].append(train_time)
        
        if save_plots:
            X_predicted, _, X_transformed = net(X_val)
            
            # plot validation data with true labels
            if d == 2 or d == 3:
                if not os.path.exists(directory + '/data'):
                    os.makedirs(directory + '/data')
                plot.plot_multiclass_toydata(X_val, C_val)
                plt.title('validation data \n for %s with %d samples' % (data['data_example'], data['V']))
                plt.savefig(directory + '/data/rep_%d.png' % i, bbox_inches='tight')
                plt.clf()
                plt.close()
            
            # plot prediction on validation data
                if not os.path.exists(directory + '/prediction'):
                    os.makedirs(directory + '/prediction')
                C_pred = prediction.pred(X_predicted)
                plot.plot_multiclass_toydata(X_val, C_pred)
                if d == 2:
                    plot.plot_multiclass_prediction(net)
                plt.title('prediction for validation data \n (%s with %d samples)' % (data['data_example'], data['V']))
                plt.savefig(directory + '/prediction/rep_%d.png' % i, bbox_inches='tight')
                plt.clf()
                plt.close()

            # plot transformation of features for validation data
            if not os.path.exists(directory + '/transformation'):
                os.makedirs(directory + '/transformation')
            # choose suitable version
            if network['d_hat'] == 2:
                dim_plot = 2
            else:
                dim_plot = 3
            if network['d_hat'] != dim_plot:
                dim_reduction = True
            else:
                dim_reduction = False
            constant_axes = True
            output_layer = True
            # plot output layer
            plot.plot_multiclass_transformation(X_transformed, C_val, dim_plot, dim_reduction, constant_axes, 
                                               output_layer=output_layer, net=net, show_output=True)
            plt.savefig(directory + '/transformation/rep_%d.png' % i, bbox_inches='tight')
            plt.clf()
            plt.close()
            # create video
            plot.plot_multiclass_transformation(X_transformed, C_val, dim_plot, dim_reduction, constant_axes, 
                                               output_layer=output_layer, net=net, 
                                               save=directory + '/transformation/rep_%d' % i)
            
            # plot trajectories for a subset of validation data
            if not os.path.exists(directory + '/trajectories'):
                os.makedirs(directory + '/trajectories')
            choose_data = np.random.choice(data['V'], 50)
            # plot complete trajectories
            plot.plot_multiclass_trajectories(X_transformed[choose_data, :, :], C_val[choose_data,:], 
                                   dim_plot, show_output=True)
            plt.savefig(directory + '/trajectories/rep_%d.png' % i, bbox_inches='tight')
            plt.clf()
            plt.close()
            # create video
            plot.plot_multiclass_trajectories(X_transformed[choose_data, :, :], C_val[choose_data,:], 
                                   dim_plot, save=directory + '/trajectories/rep_%d' % i)
    
    results['train_time']['avg_time'] = statistics.mean(results['train_time']['rep_time'])
    
    if save_results:
        # store results in seperate files
        with open(directory + '/configuration.json', 'w') as configuration_file:
            json.dump(results['configuration'], configuration_file)
        with open(directory + '/network_info.json', 'w') as network_info_file:
            json.dump(results['network_info'], network_info_file)
        with open(directory + '/statistics.json', 'w') as statistics_file:
            json.dump(results['statistics'], statistics_file)
        with open(directory + '/train_time.json', 'w') as train_time_file:
            json.dump(results['train_time'], train_time_file)
            
    if save_statistics:
        # create plots
        cost = {'training cost': results['statistics']['cost_train'], 
                'validation cost': results['statistics']['cost_val']}
        acc = {'training accuracy': results['statistics']['acc_train'], 
               'validation accuracy': results['statistics']['acc_val']}
        visualization.plot_stats(cost, 'cost')
        plt.savefig(directory + '/cost_std.png', bbox_inches='tight')
        plt.clf()
        plt.close()
        visualization.plot_stats(cost, 'cost', include_std=False, include_reps=True)
        plt.savefig(directory + '/cost_reps.png', bbox_inches='tight')
        plt.clf()
        plt.close()
        visualization.plot_stats(acc, 'acc')
        plt.savefig(directory + '/acc_std.png', bbox_inches='tight')
        plt.clf()
        plt.close()
        visualization.plot_stats(acc, 'acc', include_std=False, include_reps=True)
        plt.savefig(directory + '/acc_reps.png', bbox_inches='tight')
        plt.clf()
        plt.close()
        
        # create tables
        t_cost = visualization.table_stats(cost, 'cost', include_reps=True)
        t_cost_string = t_cost.get_string()
        t_acc = visualization.table_stats(acc, 'acc', include_reps=True)
        t_acc_string = t_acc.get_string()
        with open(directory + '/stats_table.txt', 'w') as file:
            file.write(t_cost_string + '\n')
            file.write(t_acc_string)
        
        
        
