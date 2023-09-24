import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from random import random


"""
    changes: 
           In this modified version, the plot_instance_time_domain function determines the number of subplots required based on
           the number of columns in the DataFrame. It then creates the subplots using plt.subplots(num_subplots, 1
"""
def plot_metrics(history):
    # Extract the training and validation metrics
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    train_accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']

    # Create subplots
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8, 12))

    # Plot the loss curves
    axes[0].plot(train_loss, label='Training Loss')
    axes[0].plot(val_loss, label='Validation Loss')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()

    # Plot the accuracy curves
    axes[1].plot(train_accuracy, label='Training Accuracy')
    axes[1].plot(val_accuracy, label='Validation Accuracy')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()

    # Adjust spacing between subplots
    plt.subplots_adjust(hspace=0.4)

    # Display the plot
    plt.show()
    
    
def plot_instance_time_domain(df):
    """Visualizes the movement instance to a plot in time domain.

    Args:
        df: The DataFrame to be visualized in time domain.

    Returns:

    """
    num_subplots = len(df.columns) // 3 + (len(df.columns) % 3 > 0)
    fig, axes = plt.subplots(num_subplots, 1, figsize=(13, 6*num_subplots), sharex=True)
    
    # make sure axes iterable
    axes = axes if hasattr(axes, '__iter__') else [axes]

    column_groups = np.array_split(df.columns, num_subplots)

    for i, ax in enumerate(axes):
        start_idx = i * 3
        end_idx = start_idx + 3
        group = column_groups[i]

        df[group].plot(ax=ax, linewidth=1.5, fontsize=15)

        ax.set_xlabel('Sample', fontsize=15)
        ax.set_ylabel('Axes', fontsize=15)
        # ax.set_title(f'Group {i+1}', fontsize=14)

    plt.tight_layout()


def plot_instance_3d(df):
    """Plots pairs of 3-axes DataFrame in 3D graph.

    Args:
        df: The DataFrame to be plotted in 3D.

    Returns:

    """
    num_subplots = len(df.columns) // 3  # Calculate the number of subplots based on the number of columns

    fig = plt.figure(figsize=(6 * num_subplots, 6))
    
    for i in range(num_subplots):
        ax = fig.add_subplot(1, num_subplots, i + 1, projection='3d')

        start_index = i * 3
        end_index = start_index + 3

        xs = df.iloc[:, start_index]
        ys = df.iloc[:, start_index + 1]
        zs = df.iloc[:, start_index + 2]
        
        # Generate random color for each subplot
        color = (random(), random(), random())

        ax.scatter(xs, ys, zs, color=color, s=50, alpha=0.6, edgecolors='w')

        ax.set_xlabel(df.columns[start_index])
        ax.set_ylabel(df.columns[start_index + 1])
        ax.set_zlabel(df.columns[start_index + 2])

    plt.show()


def plot_np_instance(np_array, columns_list):
    """Plot NumPy instance using DataFrames (pandas). It first transforms the array into
    DataFrame with its corresponding columns naming, and then, it plots the DataFrame in
    time domain.

    Args:
        np_array: The NumPy array to be transformed.
        columns_list: The columns list that the DataFrame and the plot will have.

    Returns:

    """
    df = pd.DataFrame(np_array, columns=columns_list)
    df.plot(figsize=(20, 10), linewidth=2, fontsize=20)
    plt.xlabel('Sample', fontsize=20)
    plt.ylabel('Axes', fontsize=20)


def plot_heatmap(df):
    """Visualizes the heatmap of the DataFrame's values.

    Args:
        df: A DataFrame.

    Returns:

    """
    plt.figure(figsize=(14, 6))
    sns.heatmap(df, cmap='plasma')


def plot_scatter_pca(df, c_name, cmap_set="plasma"):
    """Visualizes the values of the component columns of the DataFrame according to its column
    that includes the labels.

    Args:
        df: The DataFrame that contains the transformed data after the PCA procedure.
        c_name: The name of the column that includes the labels.
        cmap_set: The format of the plot.

    Returns:

    """
    if len(df.columns) == 3:
        plt.style.use('seaborn-v0_8')
        plt.figure(figsize=(16, 8))
        scatter = plt.scatter(df.iloc[:, 0], df.iloc[:, 1], c=df[c_name], cmap=cmap_set)
        plt.xlabel('First principal component')
        plt.ylabel('Second Principal Component')
        plt.legend(*scatter.legend_elements(), title=c_name)
    elif len(df.columns) == 4:
        plt.style.use('seaborn-v0_8')
        fig = plt.figure(figsize=(16, 8))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(df.iloc[:, 0], df.iloc[:, 1], df.iloc[:, 2], c=df[c_name], cmap=cmap_set)
        ax.set_xlabel('First principal component')
        ax.set_ylabel('Second Principal Component')
        ax.set_zlabel('Third Principal Component')
        ax.legend(*scatter.legend_elements(), title=c_name)
        
