from time import time
import pandas as pd
from sklearn import metrics
from sklearn.base import ClusterMixin
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from yellowbrick.cluster import KElbowVisualizer, InterclusterDistance, SilhouetteVisualizer
import seaborn as sns

import data

SEED = 42
STATS_FOLDER = 'stats/clustering'
PLOTS_FOLDER = 'plots/clustering'


# COMMON

def plot_cluster_centers(estimator, dataset, version):
    if dataset == 'fashion' and version == 'base':
        plot_cluster_centers_fashion(estimator, dataset, version)
    if dataset == 'wine' and version == 'base':
        plot_cluster_centers_wine(estimator, dataset, version)


def plot_cluster_centers_fashion(estimator, dataset, version):
    fig, ax = plt.subplots(1, estimator.n_clusters)
    centers = estimator.cluster_centers_.reshape(estimator.n_clusters, 28, 28)
    for axi, center in zip(ax.flat, centers):
        axi.set(xticks=[], yticks=[])
        axi.imshow(center, interpolation='nearest', cmap='binary')
    plt.savefig(f'{PLOTS_FOLDER}/{dataset}/{version}/{dataset}_{version}_{estimator.__class__.__name__}_cluster_centers_k{estimator.n_clusters}.png')
    plt.clf()


def plot_cluster_centers_wine(estimator, dataset, version):
    df = pd.DataFrame(estimator.cluster_centers_, columns=data.DATA[dataset][version]['x_train'].columns)
    ax = sns.heatmap(df, annot=True, cmap='Blues')
    ax.figure.subplots_adjust(bottom=0.3)
    ax.figure.savefig(f'{PLOTS_FOLDER}/{dataset}/{version}/{dataset}_{version}_{estimator.__class__.__name__}_cluster_centers_k{estimator.n_clusters}.png')
    plt.clf()


def plot_cluster_distances(estimator, dataset, version):
    visualizer = InterclusterDistance(estimator)
    visualizer.fit(data.DATA[dataset][version]['x_train'])
    visualizer.show(f'{PLOTS_FOLDER}/{dataset}/{version}/{dataset}_{version}_{estimator.__class__.__name__}_cluster_distances_k{estimator.n_clusters}.png')
    plt.clf()


def plot_cluster_silhouette(estimator, dataset, version):
    visualizer = SilhouetteVisualizer(estimator, colors='yellowbrick')
    visualizer.fit(data.DATA[dataset][version]['x_train'])
    visualizer.show(f'{PLOTS_FOLDER}/{dataset}/{version}/{dataset}_{version}_{estimator.__class__.__name__}_cluster_silhouettes_k{estimator.n_clusters}.png')
    plt.clf()


def plot_elbow(estimator, k_values, dataset, version, metric='distortion'):
    visualizer = KElbowVisualizer(estimator, k=k_values, metric=metric)
    visualizer.fit(data.DATA[dataset][version]['x_train'])
    visualizer.show(f'{PLOTS_FOLDER}/{dataset}/{version}/{dataset}_{version}_{estimator.__class__.__name__}_elbow_{metric}.png')
    plt.clf()


# KMEANS

def kmeans_fit(estimator, data, labels):
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


def kmeans_kselection(dataset, version):
    stats = []
    k_values = [2, 3, 4, 5, 6, 7, 8, 9, 10]

    for k in k_values:
        print(f'Analyzing {dataset} ({version} version) with KMeans (k={k})')
        kmeans = KMeans(n_clusters=k, random_state=SEED)
        bench = kmeans_fit(kmeans, data.DATA[dataset][version]['x_train'], data.DATA[dataset][version]['y_train'])
        stats.append(bench)
        plot_cluster_centers(kmeans, dataset, version)
        plot_cluster_distances(kmeans, dataset, version)
        plot_cluster_silhouette(kmeans, dataset, version)

    print('running elbow method')
    estimator = KMeans(random_state=SEED)
    plot_elbow(estimator, k_values, dataset, version, metric='distortion')

    stats_df = pd.DataFrame(stats).set_index('k')
    stats_df.to_csv(f'{STATS_FOLDER}/{dataset}/{version}/{dataset}_{version}_kmeans_stats.csv')
    print(f'KMeans kselection on {dataset} ({version} version) run.')


def kmeans_evaluation(dataset, version):
    stats = pd.read_csv(f'{STATS_FOLDER}/{dataset}/{version}/{dataset}_{version}_kmeans_stats.csv', index_col='k')
    stats = stats[['homo', 'compl', 'vmeas', 'ari', 'ami']]
    stats.plot(marker='o')
    plt.title(f'Evaluation of KMeans clusters on {dataset} ({version})')
    plt.xlabel('Number of clusters')
    plt.ylabel('Score Values')
    plt.legend()
    plt.savefig(f'{PLOTS_FOLDER}/{dataset}/{version}/{dataset}_{version}_KMeans_evaluation.png')
    plt.clf()


# EM

class EM(GaussianMixture, ClusterMixin):
    # Wrapper around scikitlearn's GaussianMixture so that it follows the 'clusterer' spec

    def __init__(self, n_clusters=2, **kwargs):
        kwargs["n_components"] = n_clusters
        kwargs["random_state"] = SEED
        super(EM, self).__init__(**kwargs)
        self.n_clusters = n_clusters
        self.labels_ = None
        self.cluster_centers_ = None
        self._estimator_type = 'clusterer'

    def set_params(self, n_clusters=2, **kwargs):
        kwargs["n_components"] = n_clusters
        super(EM, self).__init__(**kwargs)

    def fit(self, x, **kwargs):
        super(EM, self).fit(x)
        self.labels_ = self.predict(x)
        self.cluster_centers_ = self.means_
        return self


def em_fit(estimator, data, labels):
    t0 = time()
    estimator.fit(data)
    return {
        'k': estimator.n_clusters,
        'eta': (time() - t0),
        'homo': metrics.homogeneity_score(labels, estimator.labels_),
        'compl': metrics.completeness_score(labels, estimator.labels_),
        'vmeas': metrics.v_measure_score(labels, estimator.labels_),
        'ari': metrics.adjusted_rand_score(labels, estimator.labels_),
        'ami': metrics.adjusted_mutual_info_score(labels, estimator.labels_),
        'silhouette': metrics.silhouette_score(data, estimator.labels_, metric='euclidean')
    }


def em_kselection(dataset, version):
    stats = []
    k_values = [2, 3, 4, 5, 6, 7, 8, 9, 10]

    for k in k_values:
        print(f'Analyzing {dataset} ({version} version) with EM (k={k})')
        em = EM(n_clusters=k, random_state=SEED)
        bench = em_fit(em, data.DATA[dataset][version]['x_train'], data.DATA[dataset][version]['y_train'])
        stats.append(bench)
        plot_cluster_centers(em, dataset, version)
        plot_cluster_distances(em, dataset, version)
        plot_cluster_silhouette(em, dataset, version)

    print('running elbow method')
    estimator = EM(random_state=SEED)
    plot_elbow(estimator, k_values, dataset, version, metric='silhouette')

    stats_df = pd.DataFrame(stats).set_index('k')
    stats_df.to_csv(f'{STATS_FOLDER}/{dataset}/{version}/{dataset}_{version}_em_stats.csv')
    print(f'EM kselection on {dataset} ({version} version) run.')


def em_evaluation(dataset, version):
    stats = pd.read_csv(f'{STATS_FOLDER}/{dataset}/{version}/{dataset}_{version}_em_stats.csv', index_col='k')
    stats = stats[['homo', 'compl', 'vmeas', 'ari', 'ami']]
    stats.plot(marker='o')
    plt.title(f'Evaluation of EM clusters on {dataset} ({version})')
    plt.xlabel('Number of clusters')
    plt.ylabel('Score Values')
    plt.legend()
    plt.savefig(f'{PLOTS_FOLDER}/{dataset}/{version}/{dataset}_{version}_EM_evaluation.png')
    plt.clf()


# MAIN

if __name__ == '__main__':

    dataset_to_run = 'fashion'
    version_to_run = 'base'
    data.load_data(dataset_to_run, version_to_run)
    kmeans_kselection(dataset_to_run, version_to_run)
    kmeans_evaluation(dataset_to_run, version_to_run)
    em_kselection(dataset_to_run, version_to_run)
    em_evaluation(dataset_to_run, version_to_run)

    dataset_to_run = 'wine'
    version_to_run = 'base'
    data.load_data(dataset_to_run, version_to_run)
    kmeans_kselection(dataset_to_run, version_to_run)
    kmeans_evaluation(dataset_to_run, version_to_run)
    em_kselection(dataset_to_run, version_to_run)
    em_evaluation(dataset_to_run, version_to_run)

    print('Clustering run.')
