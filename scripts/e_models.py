import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from imblearn.under_sampling import RandomUnderSampler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, learning_curve
from sklearn.preprocessing import StandardScaler
from tensorflow import keras


# Configuration for reproducible results.
np.random.seed(9)
tf.random.set_seed(9)
random.seed(9)
os.environ['PYTHONHASHSEED'] = str(9)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

DATASET_PATH = '/media/kryekuzhinieri/Thesis/Datasets 2/Stress Recognition In Automobile Drivers/csv_files/preprocessed_data/final_data/all_drives.csv'


def get_train_test_data(path, undersample=False, shuffle=False, test_drives=["Drive15", "Drive16"]):
    """
    Splits the data into training and testing.
    inputs:
        path - (string) path to processed data.
        undersample: (bool) whether to balance the response. Default True
        shuffle: (bool) whether to shuffle the data randomly. Default True
        test_drives: (list) list of drives to select as test. Default Drive15, Drive16
    """
    data = pd.read_csv(path)
    data = data.dropna()
    X_train = data[~data["Drive"].isin(test_drives)]
    X_test = data[data["Drive"].isin(test_drives)]
    if shuffle:
        X_train = X_train.sample(frac=1).reset_index(drop=True)
        X_test = X_test.sample(frac=1).reset_index(drop=True)
    y_train = X_train["Stress_mean"]
    y_test = X_test["Stress_mean"]
    X_train = X_train.drop(["time", "Drive", "Stress_mean"], axis=1)
    X_test = X_test.drop(["time", "Drive", "Stress_mean"], axis=1)
    if undersample:
        undersample = RandomUnderSampler(sampling_strategy="majority")
        X_train, y_train = undersample.fit_resample(X_train, y_train)
        X_test, y_test = undersample.fit_resample(X_test, y_test)
    print("Train Data:", len(X_train), len(y_train))
    print("Test Data:", len(X_test), len(y_test))
    return X_train, y_train, X_test, y_test


def plot_dimension_reduction_technique(y, results, title):
    """
    Plots dimension reduction such as PCA or LDA.
    inputs:
        y: (Pandas DataFrame or Numpy Array) y_train data.
        x: (Pandas DataFrame or Numpy Array) X_train data.
        title: (string) title of the plot.
    """
    colors = ['red', 'green', 'blue']
    target_names = [1.0, 3.0, 5.0]
    plt.figure()
    for index, color, target in zip([1.0, 3.0, 5.0], colors, target_names):
        plt.scatter(
            results[y == index, 0],
            results[y == index, 1],
            alpha=0.8,
            color=color,
            label=target
        )
    plt.title(f'Dimension Reduction - {title}', fontsize=18)
    plt.xlabel('Component I')
    plt.ylabel('Component II')
    plt.legend(loc='best', shadow=False, scatterpoints=1)
    if title == "Principal Component Analysis":
        plt.xlim([0, 50])
    plt.show()


def plot_grid_search(data):
    """
    Plots grid search most important features.
    inputs:
        data: (Pandas DataFrame) Results of grid_search.cv_results_
    """
    columns = [c for c in data.columns if c not in ["mean_test_score", "rank_test_score"]]
    fig, axs = plt.subplots(6, figsize=(12, 20), constrained_layout=True)
    colors = [
        'lightgrey', 'coral', 'lightgreen', 'wheat', 'lightpink', 'skyblue',
    ]
    for i, col in enumerate(columns):
        sns.barplot(
            x=col,
            y='mean_test_score',
            data=data,
            ax=axs[i],
            color=colors[i],
            orient='v'
        )
        axs[i].set_title(label=col.title().replace("_", " "), size=15, weight='bold')
        axs[i].set_xlabel(xlabel="", fontsize=12)
        axs[i].set_ylabel(ylabel="Mean Test Score", fontsize=12)

    plt.show()


def plot_learning_curve(model, X_train, y_train, title):
    """
    Plots learning curve of model.
    inputs:
        model: (keras) deep learning model.
        X_train: (Pandas DataFrame or Numpy Array) training data.
        y_train: (Pandas DataFrame or Numpy Array) test data.
        title: (string) title of the plot.
    """
    train_sizes, train_scores, test_scores = learning_curve(
        estimator=model,
        X=X_train,
        y=y_train,
        cv=10,
        train_sizes=np.linspace(0.1, 1.0, 30),
        n_jobs=-1
    )
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    fig = plt.figure()
    plt.plot(train_sizes, train_mean, color='red', marker='o', markersize=5, label='Training Accuracy')
    plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.15, color='red')
    plt.plot(train_sizes, test_mean, color='green', marker='o', markersize=5, linestyle='--', label='Cross-validation Accuracy')
    plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')
    plt.title(f'Random Forests {title}- Learning Curve')
    plt.xlabel('Training Examples')
    plt.ylabel('Model Accuracy')
    plt.grid()
    plt.legend(loc='lower right')
    plt.show()


def get_lda(X_train, X_test, y_train):
    """
    Performs grid search to find best lda parameters.
    inputs:
        X_train: (Pandas DataFrame or Numpy Array) training data.
        X_test: (Pandas DataFrame or Numpy Array) test training data.
        y_train: (Pandas DataFrame or Numpy Array) response training data.
    """
    model = LinearDiscriminantAnalysis()
    options = {'solver': ['svd', 'lsqr', 'eigen'], }
    grid_search = GridSearchCV(model, options, scoring='accuracy', cv=10, n_jobs=-1, verbose=0)
    results = grid_search.fit(X_train, y_train)
    # print(f"Mean Accuracy is {results.best_score_}")
    # print(f"Best Model is {results.best_params_}")
    X_train = results.transform(X_train)
    X_test = results.transform(X_test)
    plot_dimension_reduction_technique(y_train, X_train, "Linear Discriminant Analysis")
    return X_train, X_test


def get_pca(X_train, X_test, y_train):
    """
    Performs Principal Component Analysis for the components which describe 95% of the variance.
    inputs:
        X_train: (Pandas DataFrame or Numpy Array) training data.
        X_test: (Pandas DataFrame or Numpy Array) test training data.
        y_train: (Pandas DataFrame or Numpy Array) response training data.
    """
    pca = PCA(n_components=0.95)
    pca.fit(X_train)
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)
    # print(pca.explained_variance_ratio_)
    plot_dimension_reduction_technique(y_train, X_train_pca, title='Principal Component Analysis')
    return X_train_pca, X_test_pca


def get_random_forest_features(X, y):
    """
    Finds best random forest parameters.
    inputs:
        X: (Pandas DataFrame or Numpy Array) training data.
        y: (Pandas DataFrame or Numpy Array) response training data.
    """
    rf = RandomForestClassifier()
    param_grid = {
        'bootstrap': [True],
        'max_depth': [50, 60, 70, 80, 90, 100, 110],
        'max_features': [2, 3, 4],
        'min_samples_leaf': [3, 4, 5, 7, 9],
        'min_samples_split': [2, 4, 8, 10, 12],
        'n_estimators': [100, 200, 300, 1000],
    }
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=10,
        n_jobs=-1,
        verbose=10
    )
    grid_search.fit(X, y)
    print(f"Best params are: {grid_search.best_params_}")
    print(f"Mean Accuracy is {grid_search.best_score_}")
    best_grid = grid_search.best_estimator_
    most_important_features = list(X.columns[np.argsort(best_grid.feature_importances_)[-5:]])
    print(f"Most Important Features Are: {most_important_features}")
    columns_to_drop = [
        'mean_fit_time', 'std_fit_time', 'mean_score_time',
        'std_score_time', 'params', 'std_test_score'
    ]
    results = pd.DataFrame(grid_search.cv_results_).drop(columns_to_drop, axis=1)
    results = results.drop(results.filter(regex="^split").columns, axis=1).sort_values('rank_test_score').reset_index(drop=True)
    plot_grid_search(results)
    return most_important_features, grid_search.best_params_


def evaluate(model, X_test, y_test):
    """
    Calculates model accuracy.
    inputs:
        model: (keras) deep learning model.
        X_test: (Pandas DataFrame or Numpy Array) test data.
        y_test: (Pandas DataFrame or Numpy Array) test response data.
    """
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions) * 100
    print('Model Performance')
    print('Accuracy = {:0.4f}%.'.format(accuracy))
    return accuracy


def rf_classifier(X_train, y_train, X_test, y_test, best_params, title, learning_curve=True):
    """
    Builds a random forest classifier with best parameters.
    inputs:
        X_train: (Pandas DataFrame or Numpy Array) training data.
        y_train: (Pandas DataFrame or Numpy Array) response training data.
        X_test: (Pandas DataFrame or Numpy Array) test data.
        y_test: (Pandas DataFrame or Numpy Array) test response data.
        best_params: (list) names of the most important features.
        title: (string) - Title for the plot.
        learning_curve: (bool) whether to plot the learning_curve or not.
    """
    accuracy = []
    for i in range(10):
        model = RandomForestClassifier(**best_params)
        if learning_curve:
            plot_learning_curve(model, X_train, y_train, title)
        model.fit(X_train, y_train)
        result = evaluate(model, X_test, y_test)
        accuracy.append(result)
    print("Accuracy is ", np.mean(accuracy), np.std(accuracy))


def reshape_nn_data(X_train, X_test, y_train, y_test, scaling=True):
    """
    Preprocesses the data for deep learning algorithms.
    inputs:
        X_train: (Pandas DataFrame or Numpy Array) training data.
        y_train: (Pandas DataFrame or Numpy Array) response training data.
        X_test: (Pandas DataFrame or Numpy Array) test data.
        y_test: (Pandas DataFrame or Numpy Array) test response data.
        scaling: (bool) whether to scale the data or not.
    """
    y_train = y_train.apply(lambda x: 0.0 if x == 1.0 else 1.0 if x == 3.0 else 2.0)
    y_test = y_test.apply(lambda x: 0.0 if x == 1.0 else 1.0 if x == 3.0 else 2.0)
    n_cols = X_train.shape[1]
    n_rows = X_train.shape[0]
    n_rows_test = X_test.shape[0]
    if not isinstance(X_train, (np.ndarray, np.generic)):
        X_train = X_train.to_numpy()
        X_test = X_test.to_numpy()

    if scaling:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    X_train = X_train.reshape((n_rows, n_cols, 1))
    X_test = X_test.reshape((n_rows_test, n_cols, 1))
    y_train = keras.utils.to_categorical(y_train, num_classes=3)
    y_test = keras.utils.to_categorical(y_test, num_classes=3)
    y_train = y_train.reshape((len(y_train), 3))
    y_test = y_test.reshape((len(y_test), 3))
    print(X_train.shape, y_train.shape)
    print(X_test.shape, y_test.shape)
    return X_train, y_train, X_test, y_test


def find_best_parameters(X_train, X_test, y_train, y_test, algorithm):
    """
    Function to find the best parameters for deep learning algorithms.
    inputs:
        X_train: (Pandas DataFrame or Numpy Array) training data.
        y_train: (Pandas DataFrame or Numpy Array) response training data.
        X_test: (Pandas DataFrame or Numpy Array) test data.
        y_test: (Pandas DataFrame or Numpy Array) test response data.
        algorithm: (string) name of the deep learning algorithm (i.e: LSTM)
    """

    n_features, n_cube, n_outputs = X_train.shape[1], X_train.shape[2], y_train.shape[1]

    optimizers = ['Adam', 'RMSprop', 'SGD']
    activations = ['relu', 'sigmoid', 'tanh']
    hidden_layersz = [2, 3, 4]
    neurons = [20, 80, 160]  # [64, 32, 16] # Filters
    dropouts = [0.1, 0.2]  # [2, 3] # # Kernel size
    batch_sizes = [16, 20, 32]
    validation_splits = [0.1, 0.2]
    epochs = [20, 30]
    df_results = pd.DataFrame(
        columns=[
            "Model", "FeatureSelection", "UnderSample", "Shuffle", "Optimizer",
            "Activations", "HiddenLayers", "Neurons", "Dropout", "Batch", "ValSplit",
            "Epochs", "TrainingMean", "TrainingStd", "ValMean", "ValStd", "TestMean",
            "TestStd"
        ]
    )
    counter = 0
    for o in optimizers:
        for a in activations:
            for hid in hidden_layersz:
                for n in neurons:
                    for my_drop in dropouts:
                        for b in batch_sizes:
                            for v in validation_splits:
                                for e in epochs:
                                    training, validation, testing, l, m, h = [], [], [], [], [], []
                                    print("Optimizer: ", o, "-Activations: ", a, "Hidden Layers: ", hid, "-Neurons: ", n, "-Dropout: ", my_drop, "-Batch: ", b, "-Val Split: ", v, "-Epochs: ", e)
                                    for i in range(5):
                                        model = create_model(
                                            n_features,
                                            n_cube,
                                            n_outputs,
                                            algorithm=algorithm,
                                            activation=a,
                                            hidden_layers=hid,
                                            neurons=n,
                                            dropout=my_drop
                                        )
                                        tr, val, tst, low, medium, high = compile_model(
                                            model,
                                            X_train,
                                            X_test,
                                            y_train,
                                            y_test,
                                            optimizer=o,
                                            epochs=e,
                                            batch_size=b,
                                            verbose=0,
                                            validation_split=v,
                                            graphs=False
                                        )
                                        training.append(tr)
                                        validation.append(val)
                                        testing.append(tst)
                                    print("Training: ", round(np.mean(training) * 100, 2), round(np.std(training) * 100, 2))
                                    print("Validation: ", round(np.mean(validation) * 100, 2), round(np.std(validation) * 100, 2))
                                    print("Test: ", round(np.mean(testing) * 100, 2), round(np.std(testing) * 100, 2))
                                    counter += 1
                                    print(counter)
                                    print('-----------------------------')
                                    res = [algorithm, "autoencoder", True, False, o, a, hid, n, my_drop, b, v, e, round(np.mean(training) * 100, 2), round(np.std(training) * 100, 2), round(np.mean(validation) * 100, 2), round(np.std(validation) * 100, 2), round(np.mean(testing) * 100, 2), round(np.std(testing) * 100, 2)]
                                    df_results.loc[len(df_results)] = res
                                    df_results.to_csv(f"{algorithm}_results.csv", index=False)


def rnn(X_train, X_test, y_train, y_test):
    n_features, n_cube, n_outputs = X_train.shape[1], X_train.shape[2], y_train.shape[1]
    training, validation, testing, low, medium, high = [], [], [], [], [], []
    for i in range(10):
        model = keras.models.Sequential()
        model.add(keras.layers.SimpleRNN(units=20, dropout=0.2, return_sequences=True, activation='tanh', input_shape=(n_features, n_cube)))
        model.add(keras.layers.SimpleRNN(units=20, dropout=0.2, return_sequences=True))
        model.add(keras.layers.SimpleRNN(units=20, dropout=0.2))
        model.add(keras.layers.Dense(n_outputs, activation='softmax'))

        tr, v, te, l, m, h = compile_model(
            model,
            X_train,
            X_test,
            y_train,
            y_test,
            optimizer="adam",
            epochs=30,
            batch_size=20,
            verbose=0,
            validation_split=0.1,
            graphs=True
        )

        training.append(tr)
        validation.append(v)
        testing.append(te)
    print("Training: ", np.mean(training), np.std(training))
    print("Validation: ", np.mean(validation), np.std(validation))
    print("Testing: ", np.mean(testing), np.std(testing))


def compile_model(model, X_train, X_test, y_train, y_test, optimizer, epochs, batch_size, verbose, validation_split, graphs=False):
    model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy']
    )
    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        verbose=verbose,
        validation_split=validation_split,
    )
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
    all_classes = np.sum(cm, axis=0)
    low = cm[0, 0] / all_classes[0]
    medium = cm[1, 1] / all_classes[1]
    high = cm[2, 2] / all_classes[2]
    # evaluate model
    _, accuracy = model.evaluate(X_test, y_test, batch_size=16, verbose=0)
    training_mean = np.mean(history.history['accuracy'])
    validation_mean = np.mean(history.history['val_accuracy']) if validation_split > 0 else None

    if graphs:
        fig = plt.figure()
        plt.plot(history.history['accuracy'], label='accuracy')
        plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend(loc='lower right')
        plt.show()

        ax = plt.subplot()
        sns.heatmap(cm, annot=True, fmt='g', ax=ax)
        ax.set_xlabel('Predicted Labels')
        ax.set_ylabel('True Labels')
        ax.set_title('Confusion Matrix')
        ax.xaxis.set_ticklabels(['Low', 'Medium', 'High'])
        ax.yaxis.set_ticklabels(['Low', 'Medium', 'High'])
        plt.show()

    return training_mean, validation_mean, accuracy, low, medium, high


def create_model(n_features, n_cube, n_outputs, algorithm, activation="relu", hidden_layers=0, neurons=100, dropout=0.2):
    if algorithm in ['SimpleRNN', 'LSTM']:
        return_sequences = True if hidden_layers >= 1 else False
        values = dict(
            units=neurons,
            dropout=dropout,
            activation=activation,
            return_sequences=return_sequences,
            input_shape=(n_features, n_cube)
        )
    elif algorithm in ['Conv1D']:
        values = dict(
            filters=neurons,  # filters,
            kernel_size=dropout,  # kernel_size,
            padding='same',
            activation=activation,
            input_shape=(n_features, n_cube)
        )

    algorithm = getattr(keras.layers, algorithm)
    model = keras.models.Sequential()
    model.add(algorithm(**values))
    for i in range(hidden_layers):
        return_sequences = True if i != hidden_layers - 1 else False
        if "return_sequences" in values.keys():
            values['return_sequences'] = return_sequences
        values.pop('input_shape', None)
        model.add(algorithm(**values))
    if algorithm.__name__ == "Conv1D":
        model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(n_outputs, activation="softmax"))
    return model


class AutoEncoders(keras.Model):
    def __init__(self, output_units):
        super().__init__()
        self.encoder = keras.Sequential([
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(16, activation='relu'),
            keras.layers.Dense(8, activation='relu')
        ])

        self.decoder = keras.Sequential([
            keras.layers.Dense(16, activation='relu'),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(output_units, activation='relu'),
        ])

    def call(self, inputs):
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)
        return decoded


def get_autoencoder(X_train, X_test, y_train):
    auto_encoder = AutoEncoders(X_train.shape[1])
    auto_encoder.compile(
        loss='mae',
        metrics=['mae'],
        optimizer='adam'
    )
    history = auto_encoder.fit(
        X_train,
        X_train,
        epochs=100,
        batch_size=32,
        validation_data=(X_test, X_test)
    )
    encoder_layer = auto_encoder.get_layer('sequential')
    X_train = pd.DataFrame(encoder_layer.predict(X_train))
    X_train = X_train.add_prefix('feature_')
    X_test = pd.DataFrame(encoder_layer.predict(X_test))
    X_test = X_test.add_prefix('feature_')
    print("Train: ", X_train.shape, "Test: ", X_test.shape)
    plot_dimension_reduction_technique(y_train, X_train.to_numpy(), "AutoEncoders")
    return X_train, X_test


def main(feature_selection, classifier, undersample, shuffle, path=DATASET_PATH, title=None, algorithm="LSTM"):
    best_params = dict(
        bootstrap=True,
        max_depth=110,
        max_features=4,
        min_samples_leaf=3,
        min_samples_split=4,
        n_estimators=200
    )
    X_train, y_train, X_test, y_test = get_train_test_data(path=DATASET_PATH, undersample=undersample, shuffle=shuffle)
    if feature_selection == "lda":
        X_train, X_test = get_lda(X_train, X_test, y_train)
    elif feature_selection == "pca":
        X_train, X_test = get_pca(X_train, X_test, y_train)
    elif feature_selection == "autoencoder":
        X_train, X_test = get_autoencoder(X_train, X_test, y_train)
    elif feature_selection == "rf":
        features, best_params = get_random_forest_features(X_train, y_train)
        X_train = X_train[features]
        X_test = X_test[features]

    # Reshape and scale the data for deep learning.
    if classifier not in ["rf"]:
        X_train, y_train, X_test, y_test = reshape_nn_data(X_train, X_test, y_train, y_test, scaling=True)
        n_timesteps, n_features, n_outputs = X_train.shape[1], X_train.shape[2], y_train.shape[1]

    if classifier == "rf":
        rf_classifier(X_train, y_train, X_test, y_test, best_params, title, learning_curve=False)
    elif classifier == "rnn":
        rnn(X_train, X_test, y_train, y_test)
    elif classifier == "loop":
        find_best_parameters(X_train, X_test, y_train, y_test, algorithm=algorithm)
