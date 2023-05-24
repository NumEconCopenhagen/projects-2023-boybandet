import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from matplotlib_venn import venn2
from pandas_datareader import wb

def plot_func(df, x, y, title): # fix x variable sÃ¥ den ikke bliver fjernet til index
    """Function for plotting dataframes
    Convert a column of a dataframe into a lineplot
    
    Args:
        df: Input dataframe
        x: values for x-axis and xlabel
        y: values for y-axis and ylabel
        title: title for the figure
    
    """
    # Function that operates on data set
    fig, ax = plt.subplots(figsize=(10, 6)) # set the figure size to 10x6
    ax.plot(df.index, df[y])
    ax.set_xlabel(x) # Label x axis
    ax.set_ylabel(y) # Label y axis
    ax.set_title(title) # Title figure
    ax.tick_params(axis='x', rotation=45) # rotate x-axis labels
    plt.subplots_adjust(bottom=0.2)
    plt.show()