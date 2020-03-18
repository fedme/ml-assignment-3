import clustering
import data


def run_clustering():
    datasets = ['fashion', 'wine']
    versions = ['ica', 'pca', 'rp', 'svd']

    for dataset in datasets:
        for version in versions:
            print(f'Running clustering on {dataset} ({version} version)')
            data.load_data(dataset, version)
            clustering.kmeans_kselection(dataset, version)
            clustering.kmeans_evaluation(dataset, version)
            clustering.em_kselection(dataset, version)
            clustering.em_evaluation(dataset, version)


if __name__ == '__main__':
    run_clustering()
