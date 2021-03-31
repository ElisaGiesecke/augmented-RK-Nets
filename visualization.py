import numpy as np
import matplotlib.pyplot as plt
from prettytable import PrettyTable

# ============================================
# Statistics plot for comparing data
# ============================================

def plot_stats(all_stats, stats_type, include_mean=True, include_std=True, 
               include_reps=False):
    """Show statistics, either cost or accuracy, for training and validation or 
    of several networks in a single plot.
    
    Parameters
    ----------
    all_stats : dict
        Keys are strings specifying where the data comes from 
        (training/validation, name of network).
        Values are lists of lists with data from each repetition.
    stats_type : str
        Type of statistics, either 'cost' or 'acc'.
    include_mean : bool
        Plot mean over repetitions. Default is True.
    include_std : bool
        Plot standard deviation over repetitions as shaded area. Default is 
        True.
    include_reps : bool
        Plot statistics of each repetition with high transparency. Default is 
        False.
    """
    # colors for plotting different statistics
    colors = ['blue', 'orange', 'green', 'red', 'purple']
    assert len(all_stats) <= len(colors)
    
    for i, stats in enumerate(all_stats):
        max_epoch = max(map(len, all_stats[stats]))
        stats_data = np.array([rep+[None]*(max_epoch-len(rep)) for rep in 
                               all_stats[stats]])
        epochs = range(max_epoch)        
        if include_mean:
            stats_mean = np.array([np.array([num for num in stats_data[:,j] if
                                    num is not None]).mean() for j in epochs]) 
            plt.plot(epochs, stats_mean, c=colors[i], label=stats)
        if include_std:
            stats_std = np.array([np.array([num for num in stats_data[:,j] if
                                    num is not None]).std() for j in epochs]) 
            plt.fill_between(epochs, stats_mean - stats_std, 
                             stats_mean + stats_std, 
                             facecolor=colors[i], alpha=0.5)
        if include_reps:
            for stats_rep in stats_data:
                plt.plot(epochs, stats_rep, c=colors[i], alpha=0.2)
    plt.legend()
    plt.xlabel('epochs')
    if stats_type == 'cost':
        plt.ylabel('cost')
    elif stats_type == 'acc':
        plt.ylabel('accuracy')
        
# ============================================
# Statistics table for comparing data
# ============================================
        
def table_stats(all_stats, stats_type, include_mean=True, include_std=True, 
                include_reps=False):
    """Create a single table of statistics, either final cost or final 
    accuracy, for training and validation or of several networks.
    
    Parameters
    ----------
    all_stats : dict
        Keys are strings specifying where the data comes from 
        (training/validation, name of network).
        Values are lists of lists with data from each repetition.
    stats_type : str
        Type of statistics, either 'cost' or 'acc'.
    include_mean : bool
        Include column with mean over repetitions. Default is True.
    include_std : bool
        Include column with standard deviation over repetitions. Default is 
        True.
    include_reps : bool
        Include columns with each repetition. Default is False.
    """
    if stats_type == 'cost':
        header = 'cost '
    elif stats_type == 'acc':
        header = 'accuracy '
    
    # create table and row of headers    
    t = PrettyTable()
    field_names = ['']
    if include_mean:
        field_names.append(header + 'mean')
    if include_std:
        field_names.append(header + 'std')
    if include_reps:
        reps = len(list(all_stats.values())[0])
        for i in range(reps):
            field_names.append(header + 'rep %d' % (i + 1))
    t.field_names = field_names
    
    # add rows for statistics
    for i, stats in enumerate(all_stats):
        stats_final_data = np.array([rep[-1] for rep in all_stats[stats]])
        add_row = [stats]
        if include_mean:
            stats_final_mean = stats_final_data.mean()
            add_row.append('%.3f' % stats_final_mean)
        if include_std:
            stats_final_std = stats_final_data.std()
            add_row.append('%.3f' % stats_final_std)
        if include_reps:
            for i in range(len(stats_final_data)):
                add_row.append('%.3f' % stats_final_data[i])
        t.add_row(add_row)
            
    return t