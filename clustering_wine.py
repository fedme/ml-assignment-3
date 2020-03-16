from time import time

import pandas as pd
from sklearn import metrics
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from yellowbrick.cluster import KElbowVisualizer, InterclusterDistance, SilhouetteVisualizer
import seaborn as sns

SEED = 42
DATA_FOLDER = 'data'
STATS_FOLDER = 'stats'
PLOTS_FOLDER = 'plots'
KMEANS_PLOTS_FOLDER = f'{PLOTS_FOLDER}/kmeans/wine'
wine_x_train, wine_y_train, wine_x_test, wine_y_test = None, None, None, None


# DATA LOADING

def load_data():
    global wine_x_train, wine_y_train, wine_x_test, wine_y_test
    wine_x_train = pd.read_csv(f'{DATA_FOLDER}/wine_white_x_train.csv')
    wine_y_train = pd.read_csv(f'{DATA_FOLDER}/wine_white_y_train.csv').iloc[:, 0].to_numpy()
    wine_x_test = pd.read_csv(f'{DATA_FOLDER}/wine_white_x_test.csv')
    wine_y_test = pd.read_csv(f'{DATA_FOLDER}/wine_white_y_test.csv').iloc[:, 0].to_numpy()


# COMMON

def plot_cluster_distances(estimator):
    visualizer = InterclusterDistance(estimator)
    visualizer.fit(wine_x_train)
    visualizer.show(f'{KMEANS_PLOTS_FOLDER}/wine_{estimator.__class__.__name__}_cluster_distances_k{estimator.n_clusters}.png')
    plt.clf()


def plot_cluster_silhouette(estimator):
    visualizer = SilhouetteVisualizer(estimator, colors='yellowbrick')
    visualizer.fit(wine_x_train)
    visualizer.show(f'{KMEANS_PLOTS_FOLDER}/wine_{estimator.__class__.__name__}_cluster_silhouettes_k{estimator.n_clusters}.png')
    plt.clf()


def plot_elbow_distortion(k_values):
    estimator = KMeans(random_state=SEED)
    visualizer = KElbowVisualizer(estimator, k=k_values, metric='distortion')
    visualizer.fit(wine_x_train)  # Fit the data to the visualizer
    visualizer.show(f'{KMEANS_PLOTS_FOLDER}/wine_{estimator.__class__.__name__}_elbow_distortion.png')
    plt.clf()


def plot_elbow_silhouette(k_values):
    estimator = KMeans(random_state=SEED)
    visualizer = KElbowVisualizer(estimator, k=k_values, metric='silhouette')
    visualizer.fit(wine_x_train)  # Fit the data to the visualizer
    visualizer.show(f'{KMEANS_PLOTS_FOLDER}/wine_{estimator.__class__.__name__}_elbow_silhouette.png')
    plt.clf()


def plot_cluster_centers(estimator):
    df = pd.DataFrame(estimator.cluster_centers_, columns=wine_x_train.columns)
    ax = sns.heatmap(df, annot=True, cmap='Blues')
    ax.figure.subplots_adjust(bottom=0.3)
    ax.savefig(f'{KMEANS_PLOTS_FOLDER}/wine_{estimator.__class__.__name__}_cluster_centers_k{estimator.n_clusters}.png')
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
    k_values = [2, 3, 4, 5, 6, 7, 8, 9, 10]

    for k in k_values:
        print(f'analyzing wine with KMeans (k={k})')
        kmeans = KMeans(n_clusters=k, random_state=SEED)
        bench = bench_k_means(kmeans, wine_x_train, wine_y_train)
        stats.append(bench)
        plot_cluster_centers(kmeans)
        plot_cluster_distances(kmeans)
        plot_cluster_silhouette(kmeans)

    print('running elbow method on distortion')
    plot_elbow_distortion(k_values)

    print('running elbow method on silhouette')
    plot_elbow_silhouette(k_values)

    stats_df = pd.DataFrame(stats).set_index('k')
    stats_df.to_csv(f'{STATS_FOLDER}/kmeans_wine_stats.csv')
    print('kmeans_kselection_wine run.')


def kmeans_evaluation():
    stats = pd.read_csv(f'{STATS_FOLDER}/kmeans_wine_stats.csv', index_col='k')
    stats = stats[['homo', 'compl', 'vmeas', 'ari', 'ami']]
    stats.plot(marker='o')
    plt.title(f'Evaluation of KMeans clusters (confirming k=??)')
    plt.xlabel('Number of clusters (k=?? was chosen)')
    plt.ylabel('Score Values')
    plt.legend()
    plt.savefig(f'{KMEANS_PLOTS_FOLDER}/wine_KMeans_evaluation.png')
    plt.clf()


# TODO:
# - study relationship between features and cluster
#


# MAIN

if __name__ == '__main__':
    load_data()
    kmeans_kselection()
    kmeans_evaluation()
    print('exp run.')
