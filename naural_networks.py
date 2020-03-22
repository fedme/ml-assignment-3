from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, RandomizedSearchCV
from sklearn.neural_network import MLPClassifier
import pandas as pd
from timeit import default_timer as timer
import matplotlib.pyplot as plt
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
    print(best_params)
    print(mean_accuracy)


def measure_mlp_accuracy():

    results_accuracy_train = {}
    results_accuracy_test = {}
    results_times = {}

    # BASE
    data_version = 'base'
    print(f'[{data_version}] Init analysis')
    data.load_data('fashion', data_version)
    x_train = data.DATA['fashion'][data_version]['x_train']
    y_train = data.DATA['fashion'][data_version]['y_train']
    classifier = MLPClassifier(
        solver='sgd',
        max_iter=600,
        learning_rate='adaptive',
        hidden_layer_sizes=(30, 30),
        alpha=0.0001,
        activation='logistic',
        random_state=SEED
    )

    print(f'[{data_version}] Measuring train CV accuracy')
    accuracy = cross_val_score(classifier, x_train, y_train, cv=3, scoring='accuracy', n_jobs=-1).mean()
    results_accuracy_train[data_version] = accuracy

    print(f'[{data_version}] Measuring train time')
    start = timer()
    classifier.fit(x_train, y_train)
    train_time = timer() - start
    results_times[data_version] = train_time

    print(f'[{data_version}] Measuring test accuracy')
    x_test = data.DATA['fashion'][data_version]['x_test']
    y_test = data.DATA['fashion'][data_version]['y_test']
    results_accuracy_test[data_version] = accuracy_score(y_test, classifier.predict(x_test))


    # PCA: solver=adam, max_iter=600, hidden_layer_size=(30, 30), alpha=0.0001, activation=logistic => score 0.8625
    data_version = 'pca'
    print(f'[{data_version}] Init analysis')
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

    print(f'[{data_version}] Measuring train CV accuracy')
    accuracy = cross_val_score(classifier, x_train, y_train, cv=3, scoring='accuracy', n_jobs=-1).mean()
    results_accuracy_train[data_version] = accuracy

    print(f'[{data_version}] Measuring train time')
    start = timer()
    classifier.fit(x_train, y_train)
    train_time = timer() - start
    results_times[data_version] = train_time

    print(f'[{data_version}] Measuring test accuracy')
    x_test = data.DATA['fashion'][data_version]['x_test']
    y_test = data.DATA['fashion'][data_version]['y_test']
    results_accuracy_test[data_version] = accuracy_score(y_test, classifier.predict(x_test))


    # ICA: solver=adam, max_iter=600, hidden_layer_sizes=(50, 50), alpha=0.001, activation=relu => score=0.831
    data_version = 'ica'
    print(f'[{data_version}] Init analysis')
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

    print(f'[{data_version}] Measuring train CV accuracy')
    accuracy = cross_val_score(classifier, x_train, y_train, cv=3, scoring='accuracy', n_jobs=-1).mean()
    results_accuracy_train[data_version] = accuracy

    print(f'[{data_version}] Measuring train time')
    start = timer()
    classifier.fit(x_train, y_train)
    train_time = timer() - start
    results_times[data_version] = train_time

    print(f'[{data_version}] Measuring test accuracy')
    x_test = data.DATA['fashion'][data_version]['x_test']
    y_test = data.DATA['fashion'][data_version]['y_test']
    results_accuracy_test[data_version] = accuracy_score(y_test, classifier.predict(x_test))


    # RP:  solver=sgd, max_iter=600, hidden_layer_sizes=(50, 50), alpha=0.0001, activation=logistic => score=0.885
    data_version = 'rp'
    print(f'[{data_version}] Init analysis')
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

    print(f'[{data_version}] Measuring train CV accuracy')
    accuracy = cross_val_score(classifier, x_train, y_train, cv=3, scoring='accuracy', n_jobs=-1).mean()
    results_accuracy_train[data_version] = accuracy

    print(f'[{data_version}] Measuring train time')
    start = timer()
    classifier.fit(x_train, y_train)
    train_time = timer() - start
    results_times[data_version] = train_time

    print(f'[{data_version}] Measuring test accuracy')
    x_test = data.DATA['fashion'][data_version]['x_test']
    y_test = data.DATA['fashion'][data_version]['y_test']
    results_accuracy_test[data_version] = accuracy_score(y_test, classifier.predict(x_test))


    # SVD: solver=adam, max_iter=600, hidden_layer_sizes=(30, 30), alpha=0.001, activation=logistic => score=0.890
    data_version = 'svd'
    print(f'[{data_version}] Init analysis')
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

    print(f'[{data_version}] Measuring train CV accuracy')
    accuracy = cross_val_score(classifier, x_train, y_train, cv=3, scoring='accuracy', n_jobs=-1).mean()
    results_accuracy_train[data_version] = accuracy

    print(f'[{data_version}] Measuring train time')
    start = timer()
    classifier.fit(x_train, y_train)
    train_time = timer() - start
    results_times[data_version] = train_time

    print(f'[{data_version}] Measuring test accuracy')
    x_test = data.DATA['fashion'][data_version]['x_test']
    y_test = data.DATA['fashion'][data_version]['y_test']
    results_accuracy_test[data_version] = accuracy_score(y_test, classifier.predict(x_test))


    # AKM: solver=adam, max_iter=600, hidden_layer_sizes=(50, 50), alpha=0.001, activation=tanh, score = 0.888
    data_version = 'aug_kmeans'
    print(f'[{data_version}] Init analysis')
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

    print(f'[{data_version}] Measuring train CV accuracy')
    accuracy = cross_val_score(classifier, x_train, y_train, cv=3, scoring='accuracy', n_jobs=-1).mean()
    results_accuracy_train[data_version] = accuracy

    print(f'[{data_version}] Measuring train time')
    start = timer()
    classifier.fit(x_train, y_train)
    train_time = timer() - start
    results_times[data_version] = train_time

    print(f'[{data_version}] Measuring test accuracy')
    x_test = data.DATA['fashion'][data_version]['x_test']
    y_test = data.DATA['fashion'][data_version]['y_test']
    results_accuracy_test[data_version] = accuracy_score(y_test, classifier.predict(x_test))


    # AEM: solver=adam, max_iter=600, hidden_layer_sizes=(50, 50), alpha=0.01, activation=logistic, score=0.886
    data_version = 'aug_em'
    print(f'[{data_version}] Init analysis')
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

    print(f'[{data_version}] Measuring train CV accuracy')
    accuracy = cross_val_score(classifier, x_train, y_train, cv=3, scoring='accuracy', n_jobs=-1).mean()
    results_accuracy_train[data_version] = accuracy

    print(f'[{data_version}] Measuring train time')
    start = timer()
    classifier.fit(x_train, y_train)
    train_time = timer() - start
    results_times[data_version] = train_time

    print(f'[{data_version}] Measuring test accuracy')
    x_test = data.DATA['fashion'][data_version]['x_test']
    y_test = data.DATA['fashion'][data_version]['y_test']
    results_accuracy_test[data_version] = accuracy_score(y_test, classifier.predict(x_test))

    print(results_accuracy_train)
    print(results_accuracy_test)
    print(results_times)

    results_accuracy_df = pd.DataFrame([results_accuracy_train, results_accuracy_test])
    results_accuracy_df.to_csv(f'{STATS_FOLDER}/neural_networks/accuracy.csv', index=False)

    results_times_df = pd.DataFrame([results_times])
    results_times_df.to_csv(f'{STATS_FOLDER}/neural_networks/train_times.csv', index=False)

    print()


def plot_accuracy():
    df = pd.read_csv(f'{STATS_FOLDER}/neural_networks/accuracy.csv')
    df = df.T
    df.columns = ['train CV accuracy', 'test accuracy']
    df.plot.bar(rot=1)
    plt.title('Train and test accuracy on different versions of fashion dataset')
    plt.xlabel('dataset version')
    plt.ylabel('accuracy score')
    plt.grid()
    plt.ylim(bottom=0.6)
    plt.savefig(f'{PLOTS_FOLDER}/neural_networks/accuracy.png')
    plt.clf()


def plot_times():
    df = pd.read_csv(f'{STATS_FOLDER}/neural_networks/train_times.csv')
    df = df.T
    df.columns = ['training time']
    df.plot.bar(rot=1)
    plt.title('Training times on different versions of fashion dataset')
    plt.xlabel('dataset version')
    plt.ylabel('training time (seconds)')
    plt.grid()
    plt.ylim(bottom=0.6)
    plt.savefig(f'{PLOTS_FOLDER}/neural_networks/train_times.png')
    plt.clf()


if __name__ == '__main__':
    # find_best_mlp('pca')
    # find_best_mlp('ica')
    # find_best_mlp('rp')
    # find_best_mlp('svd')
    # find_best_mlp('aug_kmeans')
    # find_best_mlp('aug_em')

    # measure_mlp_accuracy()
    plot_accuracy()
    plot_times()

    print()
