from time import time

import pandas as pd
from sklearn import metrics
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from yellowbrick.cluster import KElbowVisualizer, InterclusterDistance, SilhouetteVisualizer

SEED = 42
DATA_FOLDER = 'data'
STATS_FOLDER = 'stats'
PLOTS_FOLDER = 'plots'
KMEANS_PLOTS_FOLDER = f'{PLOTS_FOLDER}/kmeans/census'
census_x_train, census_y_train, census_x_test, census_y_test = None, None, None, None


# DATA LOADING

def load_data():
    global census_x_train, census_y_train, census_x_test, census_y_test

    # Load csv files
    census_x_train = pd.read_csv(f'{DATA_FOLDER}/adult_census_x_train.csv')
    census_y_train = pd.read_csv(f'{DATA_FOLDER}/adult_census_y_train.csv')
    census_x_test = pd.read_csv(f'{DATA_FOLDER}/adult_census_x_test.csv')
    census_y_test = pd.read_csv(f'{DATA_FOLDER}/adult_census_y_test.csv')

    # Check dimensions
    assert census_x_train.shape[0] == census_y_train.shape[0] == 24129
    assert census_x_test.shape[0] == census_y_test.shape[0] == 6033
    assert census_x_train.shape[1] == census_x_test.shape[1] == 95
    assert census_y_train.shape[1] == census_y_test.shape[1] == 1
    print()


# COMMON

def plot_cluster_distances(estimator):
    visualizer = InterclusterDistance(estimator)
    visualizer.fit(census_x_train)
    visualizer.show(f'{KMEANS_PLOTS_FOLDER}/census_{estimator.__class__.__name__}_cluster_distances_k{estimator.n_clusters}.png')
    plt.clf()


def plot_cluster_silhouette(estimator):
    visualizer = SilhouetteVisualizer(estimator, colors='yellowbrick')
    visualizer.fit(census_x_train)
    visualizer.show(f'{KMEANS_PLOTS_FOLDER}/census_{estimator.__class__.__name__}_cluster_silhouettes_k{estimator.n_clusters}.png')
    plt.clf()


def plot_elbow_distortion(k_values):
    estimator = KMeans(random_state=SEED)
    visualizer = KElbowVisualizer(estimator, k=k_values, metric='distortion')
    visualizer.fit(census_x_train)  # Fit the data to the visualizer
    visualizer.show(f'{KMEANS_PLOTS_FOLDER}/census_{estimator.__class__.__name__}_elbow_distortion.png')
    plt.clf()


# KMEANS

def bench_k_means(estimator, data, labels):
    t0 = time()
    estimator.fit(data)
    return {
        'k': estimator.n_clusters,
        'eta': (time() - t0),
        'inertia': estimator.inertia_,
        'homo': metrics.homogeneity_score(labels, estimator.labels_),
        'compl': metrics.completeness_score(labels, estimator.labels_),
        'vmeas': metrics.v_measure_score(labels, estimator.labels_),
        'ari': metrics.adjusted_rand_score(labels, estimator.labels_),
        'ami': metrics.adjusted_mutual_info_score(labels,  estimator.labels_),
        'silhouette': metrics.silhouette_score(data, estimator.labels_, metric='euclidean')
    }


def kmeans_kselection():
    stats = []
    k_values = [2, 3, 4, 5]

    for k in k_values:
        print(f'analyzing census with KMeans (k={k})')
        kmeans = KMeans(n_clusters=k, random_state=SEED)
        bench = bench_k_means(kmeans, census_x_train, census_y_train.iloc[:, 0].to_numpy())
        stats.append(bench)
        plot_cluster_distances(kmeans)
        plot_cluster_silhouette(kmeans)

    plot_elbow_distortion(k_values)

    stats_df = pd.DataFrame(stats).set_index('k')
    stats_df.to_csv(f'{STATS_FOLDER}/kmeans_census_stats.csv')
    print('kmeans_kselection_census run.')


def kmeans_evaluation():
    stats = pd.read_csv(f'{STATS_FOLDER}/kmeans_census_stats.csv', index_col='k')
    stats = stats[['homo', 'compl', 'vmeas', 'ari', 'ami']]
    stats.plot(marker='o')
    plt.title(f'Evaluation of KMeans clusters (confirming k=??)')
    plt.xlabel('Number of clusters (k=?? was chosen)')
    plt.ylabel('Score Values')
    plt.legend()
    plt.savefig(f'{KMEANS_PLOTS_FOLDER}/census_KMeans_evaluation.png')
    plt.clf()


# MAIN

if __name__ == '__main__':
    load_data()
    kmeans_kselection()
    print('exp run.')
