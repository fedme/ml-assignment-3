import pandas as pd

DATA_FOLDER = 'data'

DATA = {
    'fashion': {
        'base': {},
        'ica': {},
        'pca': {},
        'rp': {},
        'svd': {},
        'aug_kmeans': {},
        'aug_em': {},
    },
    'wine': {
        'base': {},
        'ica': {},
        'pca': {},
        'rp': {},
        'svd': {}
    }
}


# DATA LOADING

def load_data(dataset, version):
    if dataset == 'fashion':
        load_data_fashion(version)
    if dataset == 'wine':
        load_data_wine(version)


def load_data_fashion(version):

    y_train = pd.read_csv(f'{DATA_FOLDER}/fashion_mnist_y_train.csv').iloc[:, 0].to_numpy()
    y_test = pd.read_csv(f'{DATA_FOLDER}/fashion_mnist_y_test.csv').iloc[:, 0].to_numpy()

    if version == 'base':
        DATA['fashion']['base']['x_train'] = pd.read_csv(f'{DATA_FOLDER}/fashion_mnist_x_train.csv')
        DATA['fashion']['base']['y_train'] = y_train
        DATA['fashion']['base']['x_test'] = pd.read_csv(f'{DATA_FOLDER}/fashion_mnist_x_test.csv')
        DATA['fashion']['base']['y_test'] = y_test

    if version == 'pca':
        DATA['fashion']['pca']['x_train'] = pd.read_csv(f'{DATA_FOLDER}/reduced/fashion_pca_x_train.csv')
        DATA['fashion']['pca']['y_train'] = y_train
        DATA['fashion']['pca']['x_test'] = pd.read_csv(f'{DATA_FOLDER}/reduced/fashion_pca_x_test.csv')
        DATA['fashion']['pca']['y_test'] = y_test

    if version == 'ica':
        DATA['fashion']['ica']['x_train'] = pd.read_csv(f'{DATA_FOLDER}/reduced/fashion_ica_x_train.csv')
        DATA['fashion']['ica']['y_train'] = y_train
        DATA['fashion']['ica']['x_test'] = pd.read_csv(f'{DATA_FOLDER}/reduced/fashion_ica_x_test.csv')
        DATA['fashion']['ica']['y_test'] = y_test

    if version == 'rp':
        DATA['fashion']['rp']['x_train'] = pd.read_csv(f'{DATA_FOLDER}/reduced/fashion_rp_x_train.csv')
        DATA['fashion']['rp']['y_train'] = y_train
        DATA['fashion']['rp']['x_test'] = pd.read_csv(f'{DATA_FOLDER}/reduced/fashion_rp_x_test.csv')
        DATA['fashion']['rp']['y_test'] = y_test

    if version == 'svd':
        DATA['fashion']['svd']['x_train'] = pd.read_csv(f'{DATA_FOLDER}/reduced/fashion_svd_x_train.csv')
        DATA['fashion']['svd']['y_train'] = y_train
        DATA['fashion']['svd']['x_test'] = pd.read_csv(f'{DATA_FOLDER}/reduced/fashion_svd_x_test.csv')
        DATA['fashion']['svd']['y_test'] = y_test

    if version == 'aug_kmeans':
        DATA['fashion']['aug_kmeans']['x_train'] = pd.read_csv(f'{DATA_FOLDER}/augmented/fashion_aug_kmeans_x_train.csv')
        DATA['fashion']['aug_kmeans']['y_train'] = y_train
        DATA['fashion']['aug_kmeans']['x_test'] = pd.read_csv(f'{DATA_FOLDER}/augmented/fashion_aug_kmeans_x_test.csv')
        DATA['fashion']['aug_kmeans']['y_test'] = y_test

    if version == 'aug_em':
        DATA['fashion']['aug_em']['x_train'] = pd.read_csv(f'{DATA_FOLDER}/augmented/fashion_aug_em_x_train.csv')
        DATA['fashion']['aug_em']['y_train'] = y_train
        DATA['fashion']['aug_em']['x_test'] = pd.read_csv(f'{DATA_FOLDER}/augmented/fashion_aug_em_x_test.csv')
        DATA['fashion']['aug_em']['y_test'] = y_test


def load_data_wine(version):
    y_train = pd.read_csv(f'{DATA_FOLDER}/wine_white_y_train.csv').iloc[:, 0].to_numpy()

    if version == 'base':
        DATA['wine']['base']['x_train'] = pd.read_csv(f'{DATA_FOLDER}/wine_white_x_train.csv')
        DATA['wine']['base']['y_train'] = pd.read_csv(f'{DATA_FOLDER}/wine_white_y_train.csv').iloc[:, 0].to_numpy()
        DATA['wine']['base']['x_test'] = pd.read_csv(f'{DATA_FOLDER}/wine_white_x_test.csv')
        DATA['wine']['base']['y_test'] = pd.read_csv(f'{DATA_FOLDER}/wine_white_y_test.csv').iloc[:, 0].to_numpy()

    if version == 'ica':
        DATA['wine']['ica']['x_train'] = pd.read_csv(f'{DATA_FOLDER}/reduced/wine_ica_x_train.csv')
        DATA['wine']['ica']['y_train'] = y_train

    if version == 'pca':
        DATA['wine']['pca']['x_train'] = pd.read_csv(f'{DATA_FOLDER}/reduced/wine_pca_x_train.csv')
        DATA['wine']['pca']['y_train'] = y_train

    if version == 'rp':
        DATA['wine']['rp']['x_train'] = pd.read_csv(f'{DATA_FOLDER}/reduced/wine_rp_x_train.csv')
        DATA['wine']['rp']['y_train'] = y_train

    if version == 'svd':
        DATA['wine']['svd']['x_train'] = pd.read_csv(f'{DATA_FOLDER}/reduced/wine_svd_x_train.csv')
        DATA['wine']['svd']['y_train'] = y_train


if __name__ == '__main__':
    load_data('wine', 'base')
    print()
