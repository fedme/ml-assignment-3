import pandas as pd

DATA_FOLDER = 'data'

DATA = {
    'fashion': {
        'base': {},
        'ica': {},
        'pca': {},
        'rp': {},
        'svd': {}
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
    if version == 'base':
        DATA['fashion']['base']['x_train'] = pd.read_csv(f'{DATA_FOLDER}/fashion_mnist_x_train.csv')
        DATA['fashion']['base']['y_train'] = pd.read_csv(f'{DATA_FOLDER}/fashion_mnist_y_train.csv').iloc[:, 0].to_numpy()
        DATA['fashion']['base']['x_test'] = pd.read_csv(f'{DATA_FOLDER}/fashion_mnist_x_test.csv')
        DATA['fashion']['base']['y_test'] = pd.read_csv(f'{DATA_FOLDER}/fashion_mnist_y_test.csv').iloc[:, 0].to_numpy()

    if version == 'ica':
        DATA['fashion']['ica']['x_train'] = pd.read_csv(f'{DATA_FOLDER}/fashion_ica_x_train.csv')
    if version == 'pca':
        DATA['fashion']['pca']['x_train'] = pd.read_csv(f'{DATA_FOLDER}/fashion_pca_x_train.csv')
    if version == 'rp':
        DATA['fashion']['rp']['x_train'] = pd.read_csv(f'{DATA_FOLDER}/fashion_rp_x_train.csv')
    if version == 'svd':
        DATA['fashion']['svd']['x_train'] = pd.read_csv(f'{DATA_FOLDER}/fashion_svd_x_train.csv')


def load_data_wine(version):
    if version == 'base':
        DATA['wine']['base']['x_train'] = pd.read_csv(f'{DATA_FOLDER}/wine_white_x_train.csv')
        DATA['wine']['base']['y_train'] = pd.read_csv(f'{DATA_FOLDER}/wine_white_y_train.csv').iloc[:, 0].to_numpy()
        DATA['wine']['base']['x_test'] = pd.read_csv(f'{DATA_FOLDER}/wine_white_x_test.csv')
        DATA['wine']['base']['y_test'] = pd.read_csv(f'{DATA_FOLDER}/wine_white_y_test.csv').iloc[:, 0].to_numpy()

    if version == 'ica':
        DATA['wine']['ica']['x_train'] = pd.read_csv(f'{DATA_FOLDER}/wine_ica_x_train.csv')
    if version == 'pca':
        DATA['wine']['pca']['x_train'] = pd.read_csv(f'{DATA_FOLDER}/wine_pca_x_train.csv')
    if version == 'rp':
        DATA['wine']['rp']['x_train'] = pd.read_csv(f'{DATA_FOLDER}/wine_rp_x_train.csv')
    if version == 'svd':
        DATA['wine']['svd']['x_train'] = pd.read_csv(f'{DATA_FOLDER}/wine_svd_x_train.csv')


if __name__ == '__main__':
    load_data('wine', 'base')
    print()
