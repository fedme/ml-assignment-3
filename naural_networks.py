import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
import data

SEED = 42
STATS_FOLDER = 'stats'
PLOTS_FOLDER = 'plots'


def find_best_mlp(version):
    data.load_data('fashion', version)
    x_train = data.DATA['fashion'][version]['x_train']
    y_train = data.DATA['fashion'][version]['y_train']

    params = {
        "hidden_layer_sizes": [(10, 10), (20, 20), (30, 30), (50, 50), (10, 10, 10), (30, 30, 30)],
        "activation": ["logistic", "tanh", "relu"],
        "solver": ["sgd", "adam"],
        "alpha": [0.0001, 0.001, 0.01],
        "max_iter": [600]
    }

    mlp = MLPClassifier(random_state=SEED)
    mlp_gcv = RandomizedSearchCV(mlp, params, scoring='accuracy', n_iter=100, random_state=SEED, n_jobs=-1, verbose=5)
    mlp_gcv.fit(x_train, y_train)

    # PCA: solver=adam, max_iter=600, hidden_layer_size=(30, 30), alpha=0.0001, activation=logistic => score 0.8625
    # ICA: solver=adam, max_iter=600, hidden_layer_sizes=(50, 50), alpha=0.001, activation=relu => score=0.831
    # RP:  solver=sgd, max_iter=600, hidden_layer_sizes=(50, 50), alpha=0.0001, activation=logistic => score=0.885
    # SVD: solver=adam, max_iter=600, hidden_layer_sizes=(30, 30), alpha=0.001, activation=logistic => score=0.890
    # AKM: solver=adam, max_iter=600, hidden_layer_sizes=(50, 50), alpha=0.001, activation=tanh => score = 0.888
    # AEM: solver=adam, max_iter=600, hidden_layer_sizes=(50, 50), alpha=0.01, activation=logistic => score=0.886
    best_params = mlp_gcv.best_params_

    mlp_tuned = mlp_gcv.best_estimator_

    mean_accuracy = cross_val_score(mlp_tuned, x_train, y_train, cv=5, scoring='accuracy', n_jobs=-1).mean()
    print()


def mlp():
    # AKM: solver=adam, max_iter=600, hidden_layer_sizes=(50, 50), alpha=0.001, activation=tanh, score = 0.888
    data_version = 'aug_kmeans'
    data.load_data('fashion', data_version)
    x_train = data.DATA['fashion'][data_version]['x_train']
    y_train = data.DATA['fashion'][data_version]['y_train']
    classifier = MLPClassifier(
        solver='adam',
        max_iter=600,
        hidden_layer_sizes=(50, 50),
        alpha=0.001,
        activation='tanh',
        random_state=SEED)
    accuracy = cross_val_score(classifier, x_train, y_train, cv=3, scoring='accuracy').mean()

    # AEM: solver=adam, max_iter=600, hidden_layer_sizes=(50, 50), alpha=0.01, activation=logistic, score=0.886
    data_version = 'aug_em'
    data.load_data('fashion', data_version)
    x_train = data.DATA['fashion'][data_version]['x_train']
    y_train = data.DATA['fashion'][data_version]['y_train']
    classifier = MLPClassifier(
        solver='adam',
        max_iter=600,
        hidden_layer_sizes=(50, 50),
        alpha=0.01,
        activation='logistic',
        random_state=SEED)
    accuracy = cross_val_score(classifier, x_train, y_train, cv=3, scoring='accuracy').mean()

    # PCA: solver=adam, max_iter=600, hidden_layer_size=(30, 30), alpha=0.0001, activation=logistic => score 0.8625
    data_version = 'pca'
    data.load_data('fashion', data_version)
    x_train = data.DATA['fashion'][data_version]['x_train']
    y_train = data.DATA['fashion'][data_version]['y_train']
    classifier = MLPClassifier(
        solver='adam',
        max_iter=600,
        hidden_layer_sizes=(30, 30),
        alpha=0.0001,
        activation='logistic',
        random_state=SEED)
    accuracy = cross_val_score(classifier, x_train, y_train, cv=3, scoring='accuracy').mean()

    # ICA: solver=adam, max_iter=600, hidden_layer_sizes=(50, 50), alpha=0.001, activation=relu => score=0.831
    data_version = 'ica'
    data.load_data('fashion', data_version)
    x_train = data.DATA['fashion'][data_version]['x_train']
    y_train = data.DATA['fashion'][data_version]['y_train']
    classifier = MLPClassifier(
        solver='adam',
        max_iter=600,
        hidden_layer_sizes=(50, 50),
        alpha=0.001,
        activation='relu',
        random_state=SEED)
    accuracy = cross_val_score(classifier, x_train, y_train, cv=3, scoring='accuracy').mean()

    # RP:  solver=sgd, max_iter=600, hidden_layer_sizes=(50, 50), alpha=0.0001, activation=logistic => score=0.885
    data_version = 'rp'
    data.load_data('fashion', data_version)
    x_train = data.DATA['fashion'][data_version]['x_train']
    y_train = data.DATA['fashion'][data_version]['y_train']
    classifier = MLPClassifier(
        solver='sgd',
        max_iter=600,
        hidden_layer_sizes=(50, 50),
        alpha=0.0001,
        activation='logistic',
        random_state=SEED)
    accuracy = cross_val_score(classifier, x_train, y_train, cv=3, scoring='accuracy').mean()

    # SVD: solver=adam, max_iter=600, hidden_layer_sizes=(30, 30), alpha=0.001, activation=logistic => score=0.890
    data_version = 'svd'
    data.load_data('fashion', data_version)
    x_train = data.DATA['fashion'][data_version]['x_train']
    y_train = data.DATA['fashion'][data_version]['y_train']
    classifier = MLPClassifier(
        solver='adam',
        max_iter=600,
        hidden_layer_sizes=(30, 30),
        alpha=0.001,
        activation='logistic',
        random_state=SEED)
    accuracy = cross_val_score(classifier, x_train, y_train, cv=3, scoring='accuracy').mean()

    print()


if __name__ == '__main__':
    # find_best_mlp('pca')
    # find_best_mlp('ica')
    # find_best_mlp('rp')
    # find_best_mlp('svd')
    # find_best_mlp('aug_kmeans')
    # find_best_mlp('aug_em')
    mlp()
    print
