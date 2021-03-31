import sys
import configuration

# ============================================
# Main program to run experiment in console:
# Gets configuration path and mode from command line.
# Calls configuration function (either simple or complex), 
# which calls function for running experiment with specified parameters.
# Results are saved to files in experiment directory. 
# ============================================

# get configuration file and configuration mode from command line arguments
if len(sys.argv) != 3:
    raise(RuntimeError('Wrong arguments, use python3 run.py <config_path> <config_mode>'))
config_path = sys.argv[1]
config_mode = sys.argv[2]

# configurate and run simple or complex experiment
if config_mode == 'simple':
    configuration.configurate_simple(config_path)
elif config_mode == 'complex':
    configuration.configurate_complex(config_path)
else:
    raise(RuntimeError('<config_mode> has to be either complex or simple'))
        




                                        