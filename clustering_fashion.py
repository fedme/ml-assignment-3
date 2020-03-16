from time import time

import pandas as pd
from sklearn import metrics
from sklearn.base import ClusterMixin
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import calinski_harabasz_score
from sklearn.mixture import GaussianMixture
from yellowbrick.cluster import KElbowVisualizer, InterclusterDistance, SilhouetteVisualizer

SEED = 42
DATA_FOLDER = 'data'
STATS_FOLDER = 'stats'
PLOTS_FOLDER = 'plots'
KMEANS_PLOTS_FOLDER = f'{PLOTS_FOLDER}/kmeans/fashion'
EM_PLOTS_FOLDER = f'{PLOTS_FOLDER}/em/fashion'
fashion_x_train, fashion_y_train, fashion_x_test, fashion_y_test = None, None, None, None


# DATA LOADING

def load_data():
    global fashion_x_train, fashion_y_train, fashion_x_test, fashion_y_test

    # Load csv files
    fashion_x_train = pd.read_csv(f'{DATA_FOLDER}/fashion_mnist_x_train.csv')
    fashion_y_train = pd.read_csv(f'{DATA_FOLDER}/fashion_mnist_y_train.csv').iloc[:, 0].to_numpy()
    fashion_x_test = pd.read_csv(f'{DATA_FOLDER}/fashion_mnist_x_test.csv')
    fashion_y_test = pd.read_csv(f'{DATA_FOLDER}/fashion_mnist_y_test.csv').iloc[:, 0].to_numpy()


# COMMON

def plot_cluster_centers(estimator):
    fig, ax = plt.subplots(1, estimator.n_clusters)
    centers = estimator.cluster_centers_.reshape(estimator.n_clusters, 28, 28)
    for axi, center in zip(ax.flat, centers):
        axi.set(xticks=[], yticks=[])
        axi.imshow(center, interpolation='nearest', cmap='binary')
    plt.savefig(f'{PLOTS_FOLDER}/fashion/base/fashion_{estimator.__class__.__name__}_cluster_centers_k{estimator.n_clusters}.png')
    plt.clf()


def plot_cluster_distances(estimator):
    visualizer = InterclusterDistance(estimator)
    visualizer.fit(fashion_x_train)
    visualizer.show(f'{PLOTS_FOLDER}/fashion/base/fashion_{estimator.__class__.__name__}_cluster_distances_k{estimator.n_clusters}.png')
    plt.clf()


def plot_cluster_silhouette(estimator):
    visualizer = SilhouetteVisualizer(estimator, colors='yellowbrick')
    visualizer.fit(fashion_x_train)
    visualizer.show(f'{PLOTS_FOLDER}/fashion/base/fashion_{estimator.__class__.__name__}_cluster_silhouettes_k{estimator.n_clusters}.png')
    plt.clf()


def plot_elbow(estimator, k_values, metric='distortion'):
    visualizer = KElbowVisualizer(estimator, k=k_values, metric=metric)
    visualizer.fit(fashion_x_train)
    visualizer.show(f'{PLOTS_FOLDER}/fashion/base/fashion_{estimator.__class__.__name__}_elbow_{metric}.png')
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


def kmeans_kselection():
    stats = []
    k_values = [2, 3, 4, 5, 6, 7, 8, 9, 10]

    for k in k_values:
        print(f'analyzing fashion with KMeans (k={k})')
        kmeans = KMeans(n_clusters=k, random_state=SEED)
        bench = kmeans_fit(kmeans, fashion_x_train, fashion_y_train)
        stats.append(bench)
        plot_cluster_centers(kmeans)
        plot_cluster_distances(kmeans)
        plot_cluster_silhouette(kmeans)

    print('running elbow method')
    estimator = KMeans(random_state=SEED)
    plot_elbow(estimator, k_values, metric='distortion')

    stats_df = pd.DataFrame(stats).set_index('k')
    stats_df.to_csv(f'{STATS_FOLDER}/fashion/base/kmeans_fashion_stats.csv')
    print('kmeans_kselection_fashion run.')


def kmeans_evaluation():
    stats = pd.read_csv(f'{STATS_FOLDER}/fashion/base/kmeans_fashion_stats.csv', index_col='k')
    stats = stats[['homo', 'compl', 'vmeas', 'ari', 'ami']]
    stats.plot(marker='o')
    plt.title(f'Evaluation of KMeans clusters (confirming k=5)')
    plt.xlabel('Number of clusters (k=5 was chosen)')
    plt.ylabel('Score Values')
    plt.legend()
    plt.savefig(f'{PLOTS_FOLDER}/fashion/base/fashion_KMeans_evaluation.png')
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


def em_kselection():
    stats = []
    k_values = [2, 3, 4, 5, 6, 7, 8, 9, 10]

    for k in k_values:
        print(f'analyzing fashion with EM (k={k})')
        em = EM(n_clusters=k, random_state=SEED)
        bench = em_fit(em, fashion_x_train, fashion_y_train)
        stats.append(bench)
        plot_cluster_centers(em)
        plot_cluster_distances(em)
        plot_cluster_silhouette(em)

    print('running elbow method')
    estimator = EM(random_state=SEED)
    plot_elbow(estimator, k_values, metric='silhouette')

    stats_df = pd.DataFrame(stats).set_index('k')
    stats_df.to_csv(f'{STATS_FOLDER}/fashion/base/em_fashion_stats.csv')
    print('em_kselection_fashion run.')


def em_evaluation():
    stats = pd.read_csv(f'{STATS_FOLDER}/fashion/base/em_fashion_stats.csv', index_col='k')
    stats = stats[['homo', 'compl', 'vmeas', 'ari', 'ami']]
    stats.plot(marker='o')
    plt.title(f'Evaluation of EM clusters (confirming k=??)')
    plt.xlabel('Number of clusters (k=?? was chosen)')
    plt.ylabel('Score Values')
    plt.legend()
    plt.savefig(f'{PLOTS_FOLDER}/fashion/base/fashion_EM_evaluation.png')
    plt.clf()

# MAIN

if __name__ == '__main__':
    load_data()
    #kmeans_kselection()
    #kmeans_evaluation()
    em_kselection()
    em_evaluation()
    print('exp run.')
