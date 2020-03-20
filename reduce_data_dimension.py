from sklearn.decomposition import PCA, FastICA, TruncatedSVD
import pandas as pd
from sklearn.random_projection import SparseRandomProjection

import data
from dimensionality import compute_rp_reconstruction_error

SEED = 42
DATA_FOLDER = 'data/reduced'


def fit_transform_fashion(train_or_test):
    x = data.DATA['fashion']['base'][f'x_{train_or_test}']

    # PCA
    k = 20
    pca = PCA(n_components=k, random_state=SEED)
    pca_data = pca.fit_transform(x)
    pca_data = pd.DataFrame(pca_data)
    pca_data.to_csv(f'{DATA_FOLDER}/fashion_pca_x_{train_or_test}.csv', index=False)

    # ICA
    k = 650
    ica = FastICA(n_components=k, random_state=SEED)
    ica_data = ica.fit_transform(x)
    ica_data = pd.DataFrame(ica_data)
    ica_data.to_csv(f'{DATA_FOLDER}/fashion_ica_x_{train_or_test}.csv', index=False)

    # RP
    k = 500
    rp_data = None
    reconstruction_error = float('inf')
    for seed in range(20):
        rp = SparseRandomProjection(n_components=k, random_state=seed)
        rp.fit(x)
        new_reconstruction_error = compute_rp_reconstruction_error(rp, x)
        if new_reconstruction_error < reconstruction_error:
            reconstruction_error = new_reconstruction_error
            rp_data = rp.transform(x)

    rp_data = pd.DataFrame(rp_data)
    rp_data.to_csv(f'{DATA_FOLDER}/fashion_rp_x_{train_or_test}.csv', index=False)

    # SVD
    k = 40
    svd = TruncatedSVD(n_components=k, random_state=SEED)
    svd_data = svd.fit_transform(x)
    svd_data = pd.DataFrame(svd_data)
    svd_data.to_csv(f'{DATA_FOLDER}/fashion_svd_x_{train_or_test}.csv', index=False)


def fit_transform_wine(train_or_test):
    x = data.DATA['wine']['base'][f'x_{train_or_test}']

    # PCA
    k = 5
    pca = PCA(n_components=k)
    pca_data = pca.fit_transform(x)
    pca_data = pd.DataFrame(pca_data)
    pca_data.to_csv(f'{DATA_FOLDER}/wine_pca_x_{train_or_test}.csv', index=False)

    # ICA
    k = 9
    ica = FastICA(n_components=k)
    ica_data = ica.fit_transform(x)
    ica_data = pd.DataFrame(ica_data)
    ica_data.to_csv(f'{DATA_FOLDER}/wine_ica_x_{train_or_test}.csv', index=False)

    # RP
    k = 7
    rp_data = None
    reconstruction_error = float('inf')
    for seed in range(20):
        rp = SparseRandomProjection(n_components=k, random_state=seed)
        rp.fit(x)
        new_reconstruction_error = compute_rp_reconstruction_error(rp, x)
        if new_reconstruction_error < reconstruction_error:
            reconstruction_error = new_reconstruction_error
            rp_data = rp.transform(x)

    rp_data = pd.DataFrame(rp_data)
    rp_data.to_csv(f'{DATA_FOLDER}/wine_rp_x_{train_or_test}.csv', index=False)

    # SVD
    k = 5
    svd = TruncatedSVD(n_components=k, random_state=SEED)
    svd_data = svd.fit_transform(x)
    svd_data = pd.DataFrame(svd_data)
    svd_data.to_csv(f'{DATA_FOLDER}/wine_svd_x_{train_or_test}.csv', index=False)


if __name__ == '__main__':
    data.load_data('fashion', 'base')
    data.load_data('wine', 'base')

    fit_transform_fashion('train')
    fit_transform_fashion('test')

    fit_transform_wine('train')
    fit_transform_wine('test')
