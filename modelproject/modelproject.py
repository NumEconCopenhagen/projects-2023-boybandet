from scipy import optimize
import numpy as np
from scipy import optimize
import sympy as sp
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display

# Define the function that will be called when the sliders change
def update_plot(T, alpha, delta, beta, n, s):
    # Initialize arrays for variables
    K = np.zeros(T)   # capital stock
    L = np.zeros(T)   # Labour stock
    C = np.zeros(T)     # consumption
    S = np.zeros(T)   # Savings
    r = np.zeros(T)     # interest rate
    w = np.zeros(T)     # wage rate

    # Intial guesses
    K[0] = 1
    L[0] = 1

    # Loop over time periods
    for t in range(1, T):
        # Update capital
        K[t] = (1-delta)*K[t-1] + S[t-1]

        # Update labour
        L[t] = L[t-1]*(1+n)

        # Calculate wage rate
        w[t] = (1 - alpha) * (K[t] / L[t]) ** alpha

        # Calculate interest rate
        r[t] = alpha * (L[t] / K[t]) ** (1 - alpha) - delta

        # Calculate savings
        S[t] = s*w[t]

        # Calculate consumption
        if t < T-1:
            C[t] = beta * (1 + r[t+1]) * (L[t] * w[t] + (1 - delta) * K[t] - S[t])

        # Check for negative values of capital and labour
        if K[t] < 0:
            K[t] = 0
        if L[t] < 0:
            L[t] = 0

    # Create a figure with subplots
    fig, axs = plt.subplots(2, 3, figsize=(14, 10))

    # Plot capital stock
    axs[0, 0].plot(K[1:])
    axs[0, 0].set_title('Capital Stock')
    axs[0, 0].set_xlabel('Time Periods')
    axs[0, 0].set_ylabel('Capital Stock')

    # Plot wage rate
    axs[0, 1].plot(w[1:])
    axs[0, 1].set_title('Wage Rate')
    axs[0, 1].set_xlabel('Time Periods')
    axs[0, 1].set_ylabel('Wage Rate')

    # Plot interest rate
    axs[0, 2].plot(r[1:])
    axs[0, 2].set_title('Interest Rate')
    axs[0, 2].set_xlabel('Time Periods')
    axs[0, 2].set_ylabel('Interest Rate')

    # Plot labour stock
    axs[1, 0].plot(L[1:])
    axs[1, 0].set_title('Labour Stock')
    axs[1, 0].set_xlabel('Time Periods')
    axs[1, 0].set_ylabel('Labour Stock')

    # Plot consumption
    axs[1, 1].plot(C[1:99])
    axs[1, 1].set_title('Consumption')
    axs[1, 1].set_xlabel('Time Periods')
    axs[1, 1].set_ylabel('Consumption')

    # Plot consumption
    axs[1, 2].plot(S[1:])
    axs[1, 2].set_title('Savings')
    axs[1, 2].set_xlabel('Time Periods')
    axs[1, 2].set_ylabel('Savings')
