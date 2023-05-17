from gc import callbacks
from logging import INFO
import string
from typing import Tuple
import numpy as np
import tensorflow as tf

import flwr as fl
from datetime import datetime

from flwr.common.logger import log
from datetime import date

from mak.hpo import es,reduce_lr, CSVLoggerWithLr

class FlwrClient(fl.client.NumPyClient):
    """Flower NumPy Client implementing Fashion-MNIST image classification."""
    def __init__(
        self,
        model: tf.keras.Model,
        xy_train: Tuple[np.ndarray, np.ndarray],
        xy_test: Tuple[np.ndarray, np.ndarray],
        epochs: int,
        batch_size: int,
        hpo: bool,
        client_name: string,
        file_path = None,
        save_train_res = False,
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
        self.save_train_res = save_train_res
        self.callbacks = []

    def get_parameters(self,config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        r = self.model.fit(self.x_train, self.y_train, epochs=self.epochs, validation_split=0.15, 
                           verbose=1,callbacks=self.get_callbacks(int(config['round'])))
        hist = r.history
        return self.model.get_weights(), len(self.x_train), {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        print("Inside evalvate FashionMNistClient")
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test, verbose=1)
        print("Eval accuracy on Client {} : {}".format(self.client_name,accuracy))
        return loss, len(self.x_test), {"accuracy": accuracy}
    
    def get_callbacks(self,server_round : int):
        if self.save_train_res == True:
            self.callbacks.append(CSVLoggerWithLr(filename=self.file_path,append=True,server_round=server_round))
        if self.hpo == True:
            log(INFO,f"+++ Running With HPO +++ Round : {server_round}")
            self.callbacks = [es,reduce_lr,CSVLoggerWithLr(filename=self.file_path,append=True,server_round=server_round)]
        return self.callbacks
            
