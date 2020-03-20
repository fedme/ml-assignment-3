from sklearn.cluster import KMeans
import pandas as pd
from sklearn.ensemble import VotingClassifier
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

import data

SEED = 42
DATA_FOLDER = 'data/augmented'


def add_clusters_to_data_kmeans():
    x_train = data.DATA['fashion']['base']['x_train']
    x_test = data.DATA['fashion']['base']['x_test']

    # KMeans (k = 4) on train
    kmeans = KMeans(n_clusters=4, random_state=SEED)
    x_train_transformed = kmeans.fit_transform(x_train)
    x_train_new = pd.concat([x_train, pd.DataFrame(x_train_transformed)], axis=1)

    scaler_train = StandardScaler()
    x_train_new_scaled = scaler_train.fit_transform(x_train_new)
    pd.DataFrame(x_train_new_scaled).to_csv(f'{DATA_FOLDER}/fashion_aug_kmeans_x_train.csv')

    # KMeans (k = 4) on test
    kmeans = KMeans(n_clusters=4, random_state=SEED)
    x_test_transformed = kmeans.fit_transform(x_test, )
    x_test_new = pd.concat([x_test, pd.DataFrame(x_test_transformed)], axis=1)

    scaler_test = StandardScaler()
    x_test_new_scaled = scaler_test.fit_transform(x_test_new)
    pd.DataFrame(x_test_new_scaled).to_csv(f'{DATA_FOLDER}/fashion_aug_kmeans_x_test.csv')


def add_clusters_to_data_em():
    x_train = data.DATA['fashion']['base']['x_train']
    x_test = data.DATA['fashion']['base']['x_test']

    # GM (k = 4) on train
    gm = GaussianMixture(n_components=4, random_state=SEED)
    gm.fit(x_train)
    x_train_transformed = gm.predict_proba(x_train)
    x_train_new = pd.concat([x_train, pd.DataFrame(x_train_transformed)], axis=1)

    scaler_train = StandardScaler()
    x_train_new_scaled = scaler_train.fit_transform(x_train_new)
    pd.DataFrame(x_train_new_scaled).to_csv(f'{DATA_FOLDER}/fashion_aug_em_x_train.csv')

    # GM (k = 4) on test
    gm = GaussianMixture(n_components=4, random_state=SEED)
    gm.fit(x_test)
    x_test_transformed = gm.predict_proba(x_test)
    x_test_new = pd.concat([x_test, pd.DataFrame(x_test_transformed)], axis=1)

    scaler_test = StandardScaler()
    x_test_new_scaled = scaler_test.fit_transform(x_test_new)
    pd.DataFrame(x_test_new_scaled).to_csv(f'{DATA_FOLDER}/fashion_aug_em_x_test.csv')


if __name__ == '__main__':
    data.load_data('fashion', 'base')
    add_clusters_to_data_kmeans()
    add_clusters_to_data_em()
