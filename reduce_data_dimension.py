from sklearn.decomposition import PCA, FastICA, TruncatedSVD
import pandas as pd
from sklearn.random_projection import SparseRandomProjection

import data
from dimensionality import compute_rp_reconstruction_error

SEED = 42
DATA_FOLDER = 'data/reduced'


def fit_transform_fashion():
    x_train = data.DATA['fashion']['base']['x_train']

    # PCA
    k = 20
    pca = PCA(n_components=k, random_state=SEED)
    pca_data = pca.fit_transform(x_train)
    pca_data = pd.DataFrame(pca_data)
    pca_data.to_csv(f'{DATA_FOLDER}/fashion_pca_x_train.csv', index=False)

    # ICA
    k = 650
    ica = FastICA(n_components=k, random_state=SEED)
    ica_data = ica.fit_transform(x_train)
    ica_data = pd.DataFrame(ica_data)
    ica_data.to_csv(f'{DATA_FOLDER}/fashion_ica_x_train.csv', index=False)

    # RP
    k = 500
    rp_data = None
    reconstruction_error = float('inf')
    for seed in range(20):
        rp = SparseRandomProjection(n_components=k, random_state=seed)
        rp.fit(x_train)
        new_reconstruction_error = compute_rp_reconstruction_error(rp, x_train)
        if new_reconstruction_error < reconstruction_error:
            reconstruction_error = new_reconstruction_error
            rp_data = rp.transform(x_train)

    rp_data = pd.DataFrame(rp_data)
    rp_data.to_csv(f'{DATA_FOLDER}/fashion_rp_x_train.csv', index=False)

    # SVD
    k = 40
    svd = TruncatedSVD(n_components=k, random_state=SEED)
    svd_data = svd.fit_transform(x_train)
    svd_data = pd.DataFrame(svd_data)
    svd_data.to_csv(f'{DATA_FOLDER}/fashion_svd_x_train.csv', index=False)


def fit_transform_wine():
    x_train = data.DATA['wine']['base']['x_train']

    # PCA
    k = 5
    pca = PCA(n_components=k)
    pca_data = pca.fit_transform(x_train)
    pca_data = pd.DataFrame(pca_data)
    pca_data.to_csv(f'{DATA_FOLDER}/wine_pca_x_train.csv', index=False)

    # ICA
    k = 9
    ica = FastICA(n_components=k)
    ica_data = ica.fit_transform(x_train)
    ica_data = pd.DataFrame(ica_data)
    ica_data.to_csv(f'{DATA_FOLDER}/wine_ica_x_train.csv', index=False)

    # RP
    k = 7
    rp_data = None
    reconstruction_error = float('inf')
    for seed in range(20):
        rp = SparseRandomProjection(n_components=k, random_state=seed)
        rp.fit(x_train)
        new_reconstruction_error = compute_rp_reconstruction_error(rp, x_train)
        if new_reconstruction_error < reconstruction_error:
            reconstruction_error = new_reconstruction_error
            rp_data = rp.transform(x_train)

    rp_data = pd.DataFrame(rp_data)
    rp_data.to_csv(f'{DATA_FOLDER}/wine_rp_x_train.csv', index=False)

    # SVD
    k = 5
    svd = TruncatedSVD(n_components=k, random_state=SEED)
    svd_data = svd.fit_transform(x_train)
    svd_data = pd.DataFrame(svd_data)
    svd_data.to_csv(f'{DATA_FOLDER}/wine_svd_x_train.csv', index=False)


if __name__ == '__main__':
    data.load_data('fashion', 'base')
    data.load_data('wine', 'base')
    fit_transform_fashion()
    fit_transform_wine()
