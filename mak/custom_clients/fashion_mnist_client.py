from gc import callbacks
from logging import INFO
import argparse
import string
from typing import Dict, Tuple, cast
import timeit
import numpy as np
import tensorflow as tf

import flwr as fl
from flwr.common import Weights
from datetime import datetime

from flwr.common.logger import log
import os
from datetime import date

from mak.hpo import es,reduce_lr

class FashionMnistClient(fl.client.NumPyClient):
    """Flower NumPy Client implementing Fashion-MNIST image classification."""
    def __init__(
        self,
        model: tf.keras.Model,
        xy_train: Tuple[np.ndarray, np.ndarray],
        xy_test: Tuple[np.ndarray, np.ndarray],
        epochs: int,
        batch_size: int,
        hpo:int,
        client_name: string,
        file_path = None,
    ):
        tf.config.run_functions_eagerly(True)
        now = datetime.now()
        current_time = now.strftime("%H-%M-%S")
        today = date.today()
        today = str(today)
        self.model = model
        self.x_train, self.y_train = xy_train
        self.x_test, self.y_test = xy_test
        self.epochs = epochs
        self.batch_size = batch_size
        self.hpo = hpo
        self.client_name = client_name
        self.file_path = file_path
        self.callbacks = []
        if self.hpo == True:
            log(INFO,"+++ Running With HPO +++")
            filename = self.file_path
            self.callbacks = [es,reduce_lr,tf.keras.callbacks.CSVLogger(
            filename, separator=',', append=False
        )]
        else:
            filename = self.file_path
            self.callbacks = [tf.keras.callbacks.CSVLogger(filename, separator=',', append=False)]

    def get_parameters(self):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        r = self.model.fit(self.x_train, self.y_train, epochs=self.epochs, validation_data=(self.x_test, self.y_test), verbose=1,callbacks=self.callbacks)
        hist = r.history
        return self.model.get_weights(), len(self.x_train), {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test, verbose=1)
        print("Eval accuracy : ", accuracy)
        return loss, len(self.x_test), {"accuracy": accuracy}
            