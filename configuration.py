import os
import json
import experiment

# ============================================
# Configuration function for simple experiment
# ============================================

def configurate_simple(config_path):
    """Gets and structures configuration data for single experiment (simple
    configuration mode). Calls function for running experiment once with 
    parameters contained in configuration data.
    
    Parameters
    ----------
    config_path : str
        Path to configuration file for single experiment.
        
    Notes
    -----
    Example for format of configuration file can be found in 
    configs/simple_config.json
    """
    # open and load simple configuration file
    with open(config_path, 'r') as config_file:
        config = json.load(config_file)
    
    # name directory for saving results, plots and statistics    
    main_directory = 'results_of_' + os.path.splitext(os.path.basename(config_path))[0]

    # run single experiment
    print('Running single experiment with the following configuration:\n')
    print('data: ', config['data'], '\nnetwork: ', config['network'], 
          '\ntraining: ', config['training'], '\nreps: ', config['reps'],'\n')
    experiment.run_experiment(config['data'], config['network'], 
                              config['training'], reps=config['reps'], 
                              main_directory=main_directory)
    print('Done')

# ============================================
# Configuration function for complex experiment
# ============================================

def configurate_complex(config_path):
    """Gets and structures configuration data for multiple experiment (complex
    configuration mode). Calls function for running experiment multiple times
    with parameters contained in configuration data.
    
    Parameters
    ----------
    config_path : str
        Path to configuration file for multiple experiments.
        
    Notes
    -----
    Example for format of configuration file can be found in 
    configs/complex_config.json
    """
    # open and load complex configuration file
    with open(config_path, 'r') as config_file:
        config = json.load(config_file)
        
    # name directory for saving results, plots and statistics    
    main_directory = 'results_of_' + os.path.splitext(os.path.basename(config_path))[0]

    # run several experiments
    print('Running multiple experiments with the following configurations:\n')
    data_config = config['data']
    network_config = config['network']
    training_config = config['training']
    reps = config['reps']
    for data_example in data_config['data_example']:
        for N in data_config['N']:
            for V in data_config['V']:
                for binary in data_config['binary']:
                    data = {'data_example': data_example, 'N': N, 'V': V, 'binary': binary}
                    for net_architecture in network_config['net_architecture']:
                        for d_hat in network_config['d_hat']:
                            for L  in network_config['L']:
                                for act_function  in network_config['act_function']:
                                    for fill_zeros in network_config['fill_zeros']:
                                        for classifier_mode in network_config['classifier_mode']:
                                            network = {'net_architecture': net_architecture, 'd_hat': d_hat, 
                                                         'L': L, 'act_function': act_function, 
                                                         'fill_zeros': fill_zeros, 'classifier_mode': classifier_mode}
                                            for loss_criterion in training_config['loss_criterion']: 
                                                for batch_size in training_config['batch_size']:
                                                    for optim_method in training_config['optim_method']:
                                                        for weight_decay in training_config['weight_decay']:
                                                            for max_epochs in training_config['max_epochs']:
                                                                for tol in training_config['tol']:
                                                                    if optim_method == 'sgd':    
                                                                        for lr in training_config['lr']:
                                                                            for momentum in training_config['momentum']:
                                                                                training = {'loss_criterion': loss_criterion, 'batch_size': batch_size, 
                                                                                  'optim_method': optim_method, 'lr': lr, 'momentum': momentum, 
                                                                                  'weight_decay': weight_decay, 'max_epochs': max_epochs, 'tol': tol}
                                                                                print('data: ', data, '\nnetwork: ', network, '\ntraining: ', training, '\nreps: ', reps, '\n')
                                                                                experiment.run_experiment(data, network, training, reps=reps, main_directory=main_directory)
                                                                                
                                                                    else:
                                                                        lr = None
                                                                        momentum = None
                                                                        training = {'loss_criterion': loss_criterion, 'batch_size': batch_size, 
                                                                                  'optim_method': optim_method, 'lr': lr, 'momentum': momentum, 
                                                                                  'weight_decay': weight_decay, 'max_epochs': max_epochs, 'tol': tol}
                                                                        print('data: ', data, '\nnetwork: ', network, '\ntraining: ', training, '\nreps: ', reps, '\n')
                                                                        experiment.run_experiment(data, network, training, reps=reps, main_directory=main_directory)
    print('Done')

  