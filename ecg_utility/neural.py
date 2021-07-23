import tensorflow as tf
from imblearn.over_sampling import SMOTE
from tensorflow import keras

import os
import tempfile

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import seaborn as sns

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

mpl.rcParams['figure.figsize'] = (12, 10)
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']


def plot_cm(labels, predictions, p=0.5):
    """
    Plots a confusion matrix of the predicted labels on the testing data
    :param labels: True labels
    :param predictions: Model predicted labels
    :param p: Threshold for segmenting model output
    :return:
    """
    cm = confusion_matrix(labels, predictions > p)
    plt.figure(figsize=(5, 5))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title('Confusion matrix @{:.2f}'.format(p))
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')

    print('Bradycardia Detected (True Negatives): ', cm[0][0])
    print('Bradycardia Incorrectly Detected (False Positives): ', cm[0][1])
    print('False Alarm on detection (False Negatives): ', cm[1][0])
    print('Not Bradycardia detected (True Positives): ', cm[1][1])
    print('Total No Disease Diagnosis: ', np.sum(cm[1]))


class SimpleANN:
    def __init__(self, x):
        self.X = x
        self.METRICS = [
            keras.metrics.TruePositives(name='tp'),
            # keras.metrics.FalsePositives(name='fp'),
            # keras.metrics.TrueNegatives(name='tn'),
            keras.metrics.FalseNegatives(name='fn'),
            keras.metrics.BinaryAccuracy(name='accuracy'),
            # keras.metrics.Precision(name='precision'),
            # keras.metrics.Recall(name='recall'),
            keras.metrics.AUC(name='auc'),
            keras.metrics.AUC(name='prc', curve='PR'),  # precision-recall curve
        ]
        self.EPOCHS = 100
        self.BATCH_SIZE = 2048
        self.early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_prc',
            verbose=1,
            patience=10,
            mode='max',
            restore_best_weights=True)
        self.neg, self.pos = np.bincount(x['Bradycardia'])

        self.initial_bias = np.log([self.pos / self.neg])

    def preprocess_input(self):
        """
        Splitting the data into training, validation and testing sets and preprocessing the input before being passed
        to the model
        :return: The training, validation and testing features and labels (6 total)
        """
        train_df, test_df = train_test_split(self.X, test_size=0.2)
        train_df, val_df = train_test_split(train_df, test_size=0.2)

        y = self.X.Bradycardia
        X = self.X.drop(columns=['Bradycardia'])

        oversample = SMOTE()
        train_df, train_labels = oversample.fit_resample(X, y)

        train_df['Bradycardia'] = train_labels

        # Form np arrays of labels and features.
        train_labels = np.array(train_df.pop('Bradycardia'))
        bool_train_labels = train_labels != 0
        val_labels = np.array(val_df.pop('Bradycardia'))
        test_labels = np.array(test_df.pop('Bradycardia'))

        train_features = np.array(train_df)
        val_features = np.array(val_df)
        test_features = np.array(test_df)

        scaler = MinMaxScaler()
        train_features = scaler.fit_transform(train_features)
        val_features = scaler.transform(val_features)
        test_features = scaler.transform(test_features)

        train_features = np.clip(train_features, -5, 5)
        val_features = np.clip(val_features, -5, 5)
        test_features = np.clip(test_features, -5, 5)

        return train_features, val_features, test_features, train_labels, val_labels, test_labels

    def make_model(self, output_bias=None):
        """
        Model is defined and compiled
        :param output_bias:
        :return: Compiled model
        """
        train_features, val_features, test_features, _, _, _ = self.preprocess_input()
        if output_bias is not None:
            output_bias = tf.keras.initializers.Constant(output_bias)
        model = keras.Sequential([
            keras.layers.Dense(16, activation='relu', input_shape=(train_features.shape[-1],)),
            keras.layers.Dense(32, activation='tanh'),
            keras.layers.Dense(64, activation='tanh'),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(1, activation='sigmoid',
                               bias_initializer=output_bias),
        ])

        model.compile(
            optimizer=keras.optimizers.Adam(lr=1e-3),
            loss=keras.losses.BinaryCrossentropy(),
            metrics=self.METRICS)

        return model

    def get_initial_weights(self):
        """
        Saves the weights of the model compiled on the initial bias due to the imbalanced data
        :return: Initial weights of model
        """
        model = self.make_model(output_bias=self.initial_bias)
        initial_weights = os.path.join(tempfile.mkdtemp(), 'initial_weights')
        model.save_weights(initial_weights)

        return initial_weights

    def training(self):
        """
        The neural network is trained on the data while considering the class imbalance by considering class weights.
        The model then predicts on the testing set and this is compared with the true test labels and a classification
        report and confusion matrix are plotted
        :return: Prints classification report and plots a confusion matrix
        """
        train_features, val_features, test_features, train_labels, val_labels, test_labels = self.preprocess_input()
        initial_weights = self.get_initial_weights()
        total = self.neg + self.pos
        weight_for_0 = (1 / self.neg) * (total / 2.0)
        weight_for_1 = (1 / self.pos) * (total / 2.0)

        class_weight = {0: weight_for_0, 1: weight_for_1}

        weighted_model = self.make_model()
        weighted_model.load_weights(initial_weights)

        weighted_history = weighted_model.fit(
            train_features,
            train_labels,
            batch_size=self.BATCH_SIZE,
            epochs=self.EPOCHS,
            # callbacks=[early_stopping],
            validation_data=(val_features, val_labels),
            # The class weights go here
            class_weight=class_weight)

        train_predictions_weighted = weighted_model.predict(train_features, batch_size=self.BATCH_SIZE)
        test_predictions_weighted = weighted_model.predict(test_features, batch_size=self.BATCH_SIZE)

        weighted_results = weighted_model.evaluate(test_features, test_labels,
                                                   batch_size=self.BATCH_SIZE, verbose=0)
        # for name, value in zip(weighted_model.metrics_names, weighted_results):
        #   print(name, ': ', value)
        # print()

        plot_cm(test_labels, test_predictions_weighted)
        print(classification_report(test_labels, test_predictions_weighted > 0.5))


