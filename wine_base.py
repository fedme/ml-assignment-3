from time import time

import pandas as pd
from sklearn import metrics
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from yellowbrick.cluster import KElbowVisualizer, InterclusterDistance, SilhouetteVisualizer

SEED = 42
DATA_FOLDER = 'data'
STATS_FOLDER = 'stats'
PLOTS_FOLDER = 'plots/wine_base'
wine_x_train, wine_y_train, wine_x_test, wine_y_test = None, None, None, None


# DATA LOADING

def preprocess_data():
    # df_red = pd.read_csv(f'{DATA_FOLDER}/winequality-red.csv', sep=';')
    df = pd.read_csv(f'{DATA_FOLDER}/winequality-white.csv', sep=';')

    # transform into binary classification
    # quality > 6 good wine
    # quality <= 6 bad wine
    df['quality'] = df['quality'] > 6

    # separate features and labels
    x = df.drop(['quality'], axis=1)
    y = df['quality']

    balance = y.sum() / y.shape[0]

    # Separate train and test data
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=SEED,
        stratify=y)

    # Scale numerical features
    scaler_train = MinMaxScaler()
    x_train.iloc[:, :] = scaler_train.fit_transform(x_train)
    scaler_test = MinMaxScaler()
    x_test.iloc[:, :] = scaler_test.fit_transform(x_test)

    # Save CSVs
    x_train.to_csv(f'{DATA_FOLDER}/wine_white_x_train.csv', index=False)
    y_train.to_csv(f'{DATA_FOLDER}/wine_white_y_train.csv', index=False)
    x_test.to_csv(f'{DATA_FOLDER}/wine_white_x_test.csv', index=False)
    y_test.to_csv(f'{DATA_FOLDER}/wine_white_y_test.csv', index=False)
    print()


def load_data():
    global wine_x_train, wine_y_train, wine_x_test, wine_y_test
    wine_x_train = pd.read_csv(f'{DATA_FOLDER}/wine_white_x_train.csv')
    wine_y_train = pd.read_csv(f'{DATA_FOLDER}/wine_white_y_train.csv').iloc[:, 0].to_numpy()
    wine_x_test = pd.read_csv(f'{DATA_FOLDER}/wine_white_x_test.csv')
    wine_y_test = pd.read_csv(f'{DATA_FOLDER}/wine_white_y_test.csv').iloc[:, 0].to_numpy()


def find_best_mlp():
    ac_mlp = MLPClassifier(random_state=SEED)
    params = {
        "hidden_layer_sizes": [(10, 10), (30, 30), (50, 50), (100, 100), (10, 10, 10), (30, 30, 30), (30, 50, 30),
                               (50, 50, 50), (50, 100, 50)],
        "activation": ["logistic", "tanh", "relu"],
        "solver": ["sgd", "adam"],
        "alpha": [0.001, 0.01],
        "max_iter": [600]
    }

    mlp_gcv = RandomizedSearchCV(ac_mlp, params, scoring='f1', n_iter=100, random_state=SEED, n_jobs=-1, verbose=5)
    mlp_gcv.fit(wine_x_train, wine_y_train)

    # solver: adam, size: (50, 50, 50), alpha: 0.001, activation: relu, max_iter 600  => score 0.5496
    best_params = mlp_gcv.best_params_

    mlp_tuned = mlp_gcv.best_estimator_

    mean_accuracy = cross_val_score(mlp_tuned, wine_x_train, wine_y_train, cv=5, scoring='accuracy', n_jobs=-1).mean()
    mean_f1 = cross_val_score(mlp_tuned, wine_x_train, wine_y_train, cv=5, scoring='f1', n_jobs=-1).mean()
    print()


def mlp():
    classifier = MLPClassifier(
        hidden_layer_sizes=(50, 50, 50),
        solver='adam',
        alpha=0.001,
        activation='relu',
        random_state=SEED,
        max_iter=1000)

    mean_accuracy = cross_val_score(classifier, wine_x_train, wine_y_train, cv=5, scoring='accuracy', n_jobs=-1).mean()
    mean_f1 = cross_val_score(classifier, wine_x_train, wine_y_train, cv=5, scoring='f1', n_jobs=-1).mean()

    classifier.fit(wine_x_train, wine_y_train)
    disp = plot_confusion_matrix(classifier, wine_x_train, wine_y_train, display_labels=['bad wine', 'good wine'],
                                 cmap=plt.cm.get_cmap('Blues'),
                                 normalize='true')
    disp.ax_.set_title(f'MLP Classifier on Wine - Normalized confusion matrix')
    plt.savefig(f'{PLOTS_FOLDER}/wine_base_mlp_confusion_matrix.png')
    plt.clf()

    print()


if __name__ == '__main__':
    # preprocess_data()
    load_data()
    # find_best_mlp()
    mlp()
