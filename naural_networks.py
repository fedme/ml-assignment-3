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
    ac_mlp = MLPClassifier(random_state=SEED)

    x_train = data.DATA['fashion'][version]['x_train']
    y_train = data.DATA['fashion'][version]['y_train']

    params = {
        "hidden_layer_sizes": [(10, 10), (20, 20), (30, 30), (50, 50), (10, 10, 10), (30, 30, 30)],
        "activation": ["logistic", "tanh", "relu"],
        "solver": ["sgd", "adam"],
        "alpha": [0.0001, 0.001, 0.01],
        "max_iter": [600]
    }

    mlp_gcv = RandomizedSearchCV(ac_mlp, params, scoring='accuracy', n_iter=100, random_state=SEED, n_jobs=-1, verbose=5)
    mlp_gcv.fit(x_train, y_train)

    # PCA: solver=adam, max_iter=600, hidden_layer_size=(30, 30), alpha=0.0001, activation=logistic => score 0.8625
    # ICA: solver=adam, max_iter=600, hidden_layer_sizes=(50, 50), alpha=0.001, activation=relu => score=0.831
    # RP:  solver=sgd, max_iter=600, hidden_layer_sizes=(50, 50), alpha=0.0001, activation=logistic => score=0.885
    # SVD: solver=adam, max_iter=600, hidden_layer_sizes=(30, 30), alpha=0.001, activation=logistic => score=0.890
    best_params = mlp_gcv.best_params_

    mlp_tuned = mlp_gcv.best_estimator_

    mean_accuracy = cross_val_score(mlp_tuned, x_train, y_train, cv=5, scoring='accuracy', n_jobs=-1).mean()
    print()


# def mlp():
#     classifier = MLPClassifier(
#         hidden_layer_sizes=(50, 50, 50),
#         solver='adam',
#         alpha=0.001,
#         activation='relu',
#         random_state=SEED,
#         max_iter=1000)
#
#     mean_accuracy = cross_val_score(classifier, wine_x_train, wine_y_train, cv=5, scoring='accuracy', n_jobs=-1).mean()
#     mean_f1 = cross_val_score(classifier, wine_x_train, wine_y_train, cv=5, scoring='f1', n_jobs=-1).mean()
#
#     classifier.fit(wine_x_train, wine_y_train)
#     disp = plot_confusion_matrix(classifier, wine_x_train, wine_y_train, display_labels=['bad wine', 'good wine'],
#                                  cmap=plt.cm.get_cmap('Blues'),
#                                  normalize='true')
#     disp.ax_.set_title(f'MLP Classifier on Wine - Normalized confusion matrix')
#     plt.savefig(f'{PLOTS_FOLDER}/wine_base_mlp_confusion_matrix.png')
#     plt.clf()
#
#     print()


if __name__ == '__main__':
    version = 'svd'
    data.load_data('fashion', version)
    find_best_mlp(version)
    print
