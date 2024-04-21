# PyTorch implementation of augmented RK Nets

This repository contains code supplementing the paper ''Classification with Runge-Kutta Networks and feature space augmentation'' (2021), see https://www.aimsciences.org/article/doi/10.3934/jcd.2021018, based on my bachelor thesis, available [here](https://github.com/ElisaGiesecke/augmented-RK-Nets/blob/main/bachelor%20thesis.pdf).

## Implementation

The code is written in Python (version 3.6.12) using PyTorch (version 1.7.1) for an efficient implementation of neural networks. 

Following the paradigm of modular programming, the functions are grouped into separate `.py` files. There are two ways how to execute them: Either by using the enclosed Jupyter Notebooks [`simulations_of_RK_nets.ipynb`](https://github.com/ElisaGiesecke/augmented-RK-Nets/blob/main/simulations_of_RK_nets.ipynb) and [`simulations_of_RK_nets_for_image_data.ipynb`](https://github.com/ElisaGiesecke/augmented-RK-Nets/blob/main/simulations_of_RK_nets_for_image_data.ipynb), or by running the main Python module [`run.py`](https://github.com/ElisaGiesecke/augmented-RK-Nets/blob/main/run.py) in the console. 

## Basic usage

### Running experiments with Jupyter Notebook

The Jupyter Notebooks [`simulations_of_RK_nets.ipynb`](https://github.com/ElisaGiesecke/augmented-RK-Nets/blob/main/simulations_of_RK_nets.ipynb) and [`simulations_of_RK_nets_for_image_data.ipynb`](https://github.com/ElisaGiesecke/augmented-RK-Nets/blob/main/simulations_of_RK_nets_for_image_data.ipynb) allow the user to gain an insight into how datasets are loaded, in which way networks are constructed and trained, and how their performance is measured. Besides that, the notebook enables the user to choose the parameters for the experiment and adjust the visualization tools on-the-fly.

The example for point classification uploaded in [`simulations_of_RK_nets.ipynb`](https://github.com/ElisaGiesecke/augmented-RK-Nets/blob/main/simulations_of_RK_nets.ipynb) shows the classification of a two dimensional donut with three classes. For that, we chose an augmented RK Net, namely `EulerNet` of width 16, depth 40 and tanh activation, which was trained with the Adam algorithm using cross-entropy loss. All numerical results and plots are displayed directly in the notebook, apart from the data transformation videos saved in the [`notebook_experiments`](https://github.com/ElisaGiesecke/augmented-RK-Nets/tree/main/notebook_experiments) directory. 

Similarly, we provide an example for image classification in [`simulations_of_RK_nets_for_image_data.ipynb`](https://github.com/ElisaGiesecke/augmented-RK-Nets/blob/main/simulations_of_RK_nets_for_image_data.ipynb). Here, we trained the `RK4Net` of width 30<sup>2</sup>, depth 100 and tanh activation with the Adam algorithm and cross-entropy loss on Fashion-MNIST. The numerical results which are not included in the notebook are saved in the [`notebook_experiments_image_data`](https://github.com/ElisaGiesecke/augmented-RK-Nets/tree/main/notebook_experiments_image_data) directory. 

### Running experiments in console

Alternatively, experiments can be carried out in the console using the main Python module [`run.py`](https://github.com/ElisaGiesecke/augmented-RK-Nets/blob/main/run.py). For that, the user needs to provide two command line arguments: The first one is the path to a `.json` file containing the configurations concerning the desired data, network architecture and training algorithm, as well as how often each experiment should be repeated. The second argument specifies the configuration mode, given either by the string `simple` or `complex`. This indicates whether the configuration file provides parameters for a single or for multiple experiments. 

Examples for the format of both types of configurations can be found in the [`configs`](https://github.com/ElisaGiesecke/augmented-RK-Nets/tree/main/configs) directory named [`simple_config.json`](https://github.com/ElisaGiesecke/augmented-RK-Nets/blob/main/configs/simple_config.json) and [`complex_config.json`](https://github.com/ElisaGiesecke/augmented-RK-Nets/blob/main/configs/complex_config.json), respectively. Thus, the command line would be given by either 
```python
python run.py configs/simple_config.json simple
```
or
```python
python run.py configs/complex_config.json complex
```

Depending on the configuration mode, the corresponding function in the module [`configuration.py`](https://github.com/ElisaGiesecke/augmented-RK-Nets/blob/main/configuration.py) is called, which in turn starts the function `run_experiment` in the file [`experiment.py`](https://github.com/ElisaGiesecke/augmented-RK-Nets/blob/main/experiment.py). This function contains the core piece of the program and is the equivalent of the Jupyter Notebook file mentioned above. It saves the experiment results in separate directories marked with a timestamp.

## Examples

To reproduce the numerical examples analyzed in the paper, we provide the respective configuration files in the [`experiments`](https://github.com/ElisaGiesecke/augmented-RK-Nets/tree/main/configs/experiments) directory within the [`configs`](https://github.com/ElisaGiesecke/augmented-RK-Nets/tree/main/configs) directory.

* Experiments on network width: [`width.json`](https://github.com/ElisaGiesecke/augmented-RK-Nets/blob/main/configs/experiments/width.json)
* Experiments on network depth: [`depth.json`](https://github.com/ElisaGiesecke/augmented-RK-Nets/blob/main/configs/experiments/depth.json)
* Experiments on network activation: [`activation.json`](https://github.com/ElisaGiesecke/augmented-RK-Nets/blob/main/configs/experiments/activation.json)
* Comparison of RK Nets to standard network: [`RKNets.json`](https://github.com/ElisaGiesecke/augmented-RK-Nets/blob/main/configs/experiments/RKNets.json) and [`StandardNet.json`](https://github.com/ElisaGiesecke/augmented-RK-Nets/blob/main/configs/experiments/StandardNet.json)
* Experiments on training method (not included in paper): [`training.json`](https://github.com/ElisaGiesecke/augmented-RK-Nets/blob/main/configs/experiments/training.json)
* Extension to image classification: all files in [`image_experiments`](https://github.com/ElisaGiesecke/augmented-RK-Nets/blob/main/configs/image_experiments)

All of these configurations require the mode `complex`.
Note that the visualization tools need to be adapted to the specific settings in order to produce the plots presented in the paper (e.g. choice of color maps and of dimensionality reduction technique).
