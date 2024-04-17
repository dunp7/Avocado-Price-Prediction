'''An py file to def Functions'''
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.linear_model import LinearRegression


def plot_acf_pacf(data, lags=None):
    """
    Plots ACF and PACF for given data.
    
    Parameters:
    - data: Time series data.
    - lags: Number of lags to consider in the plots.
    """
    # Create subplots
    fig, ax = plt.subplots(2, 1, figsize=(10, 5))
    
    # Plot ACF
    plot_acf(data, lags=lags, ax=ax[0])
    ax[0].set_title('Autocorrelation Function (ACF)')
    
    # Plot PACF
    plot_pacf(data, lags=lags, ax=ax[1])
    ax[1].set_title('Partial Autocorrelation Function (PACF)')
    
    plt.tight_layout()
    plt.show()


def lagged_scatterplot(data, lag=1, x_col='Date', y_col='Value'):
    """
    Plot a scatter plot between a lagged variable and the original variable in a time series.
    
    Parameters:
    - data: DataFrame containing the time series data.
    - lag: Lag for which to create the scatter plot (default is 1).
    - x_col: Name of the column to be plotted on the x-axis (default is 'Date').
    - y_col: Name of the column to be plotted on the y-axis (default is 'Value').
    
    Returns:
    - None (displays the scatter plot).
    """
    # Shift the original variable by the lag
    data_shifted = data[y_col].shift(-lag)
    
    # Fit linear regression line
    # model = LinearRegression()
    # X = data[y_col].values.reshape(-1, 1)
    # y = data_shifted.values.reshape(-1, 1)
    # model.fit(X[:-lag], y[:-lag])
    # y_pred = model.predict(X)
    
    # Plot 
    plt.figure(figsize=(5, 6))
    # plt.plot(data[x_col], y_pred, color='red', label='Linear Regression')
    plt.scatter(data[x_col], data_shifted, alpha=0.5)
    plt.title(f'Scatter Plot: {y_col} vs. Lagged {y_col} (lag={lag})')
    plt.xlabel(x_col)
    plt.ylabel(f'Lagged {y_col}')
    plt.grid(True)
    plt.show()