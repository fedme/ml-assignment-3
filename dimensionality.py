import numpy as np
from sklearn.decomposition import PCA, FastICA
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import data

SEED = 42
STATS_FOLDER = 'stats/dimensionality'
PLOTS_FOLDER = 'plots/dimensionality'


# COMMON

def plot_main_components(estimator, dataset):
    if dataset == 'fashion':
        plot_main_components_fashion(estimator, dataset)
    if dataset == 'wine':
        plot_main_components_wine(estimator, dataset)


def plot_main_components_fashion(estimator, dataset):
    try:
        k = estimator.n_components_
    except AttributeError:
        k = estimator.n_components

    fig, ax = plt.subplots(1, k)
    centers = estimator.components_.reshape(k, 28, 28)
    for axi, center in zip(ax.flat, centers):
        axi.set(xticks=[], yticks=[])
        axi.imshow(center, interpolation='nearest', cmap='binary')
    plt.savefig(f'{PLOTS_FOLDER}/{dataset}/{dataset}_{estimator.__class__.__name__}_main_components_k{k}.png')
    plt.clf()


def plot_main_components_wine(estimator, dataset):
    try:
        k = estimator.n_components_
    except AttributeError:
        k = estimator.n_components

    df = pd.DataFrame(estimator.components_, columns=data.DATA[dataset]['base']['x_train'].columns)
    ax = sns.heatmap(df, annot=False, cmap='Blues')
    ax.figure.subplots_adjust(bottom=0.3)
    plt.title(f'Main components of {dataset} dataset')
    ax.figure.savefig(f'{PLOTS_FOLDER}/{dataset}/{dataset}_{estimator.__class__.__name__}_main_components_k{k}.png')
    plt.clf()


# PCA

def plot_pca_explained_variance(dataset, stats_df):
    stats_df.plot()
    plt.title(f'Evaluation of PCA components on {dataset} dataset')
    plt.xlabel('Number of components')
    plt.ylabel('Explained variance ratio')
    plt.legend()
    plt.grid()
    plt.savefig(f'{PLOTS_FOLDER}/{dataset}/{dataset}_PCA_explained_variance.png')
    plt.clf()


def run_pca(dataset):
    x_train = data.DATA[dataset]['base']['x_train']

    max_k = 150 if dataset == 'fashion' else 11

    stats = []
    k_values = range(1, max_k)

    for k in k_values:
        print(f'Analyzing {dataset} with PCA (k={k})')
        pca = PCA(n_components=k)
        pca.fit(x_train)
        stats.append({
            'k': k,
            'total_explained_variance_ratio': pca.explained_variance_ratio_.sum(),
            'k_component_explained_variance_ratio': pca.explained_variance_ratio_[-1]
        })

        if k == 5:
            plot_main_components(pca, dataset)

    stats_df = pd.DataFrame(stats).set_index('k')
    plot_pca_explained_variance(dataset, stats_df)


# ICA

def plot_ica_mean_kurtosis(dataset, stats_df):
    stats_df.plot()
    plt.title(f'Evaluation of ICA components on {dataset} dataset')
    plt.xlabel('Number of components')
    plt.ylabel('Mean Squared Kurtosis')
    plt.legend()
    plt.grid()
    plt.savefig(f'{PLOTS_FOLDER}/{dataset}/{dataset}_FastICA_mean_kurtosis.png')
    plt.clf()


def run_ica(dataset):
    x_train = data.DATA[dataset]['base']['x_train']

    k_values = []

    if dataset == 'fashion':
        k_values = [2, 5, 10, 20, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700]
    if dataset == 'wine':
        k_values = range(2, 11)

    stats = []

    for k in k_values:
        print(f'Analyzing {dataset} with ICA (k={k})')
        ica = FastICA(n_components=k, random_state=SEED)

        x_train_transformed = ica.fit_transform(x_train)
        x_train_transformed = pd.DataFrame(x_train_transformed)
        mean_squared_kurtosis = x_train_transformed.kurt(axis=0).pow(2).mean()

        stats.append({
            'k': k,
            'mean_squared_kurtosis': mean_squared_kurtosis
        })

        if k == 5:
            plot_main_components(ica, dataset)

    stats_df = pd.DataFrame(stats).set_index('k')
    plot_ica_mean_kurtosis(dataset, stats_df)


# MAIN

if __name__ == '__main__':

    dataset_to_run = 'wine'

    data.load_data(dataset_to_run, 'base')

    # run_pca(dataset_to_run)

    run_ica(dataset_to_run)

    print('Dimensionality run.')
