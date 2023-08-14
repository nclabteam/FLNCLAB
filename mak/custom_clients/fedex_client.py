from collections import OrderedDict
import warnings
from mak.custom_clients.flwr_client import FlwrClient
import tensorflow as tf
from typing import Tuple
import numpy as np
import flwr as fl
from mak.fedex.hyperparameters import Hyperparameters
import tensorflow_probability as tfp
from flwr.common.logger import log
from datetime import date
from logging import INFO

from mak.hpo import es,reduce_lr, CSVLoggerWithLr

warnings.filterwarnings("ignore", category=UserWarning)

class FedExClient(fl.client.NumPyClient):
    """FedEx Client extending FlwrClient."""
    def __init__(
        self,
        config : dict,
        hpo: bool,
        model: tf.keras.Model,
        xy_train: Tuple[np.ndarray, np.ndarray],
        xy_test: Tuple[np.ndarray, np.ndarray],
        epochs: int,
        batch_size: int,
        client_name: str,
        file_path=None,
        save_train_res=False,
    ):
        self.model = model
        self.x_train, self.y_train = xy_train
        self.x_test, self.y_test = xy_test
        self.epochs = epochs
        self.batch_size = batch_size
        self.client_name = client_name
        self.file_path = file_path
        self.save_train_res = save_train_res
        self.callbacks = []
        self.hpo = hpo
        self.config = config
        self.hyperparameters = Hyperparameters(self.config['hyperparam_config_nr']) #hyperparmeter search space 
        self.hyperparameters.read_from_csv(self.config['hyperparam_file'])

    def get_parameters(self,config):
        return self.model.get_weights()


    def set_parameters_evaluate(self, parameters, config):
       self.model.set_weights(parameters)

    def set_parameters_train(self, parameters, config):
            server_round = config['round']
             # obtain hyperparams and distribution
            self.distribution = parameters[-1]
            self.hyperparameters_config, self.hidx = self._sample_hyperparams()
            # print(f"++++++++++++++ Round : {server_round} hidx : {self.hidx} opt : {self.model.optimizer.name} lr : {self.model.optimizer.learning_rate.numpy()}")
                # remove hyperparameter distribution from parameter list
            parameters = parameters[:-1]
                
            self.model.optimizer.learning_rate.assign(self.hyperparameters_config['learning_rate'])
            # self.model.optimizer.momentum.assign(self.hyperparam_config['momentum'])
            # self.model.optimizer.weight_decay.assign(self.hyperparam_config['weight_decay'])
            # self.net.dropout = self.hyperparam_config['dropout']
            self.model.set_weights(parameters)
            return parameters
    
    def fit(self, parameters, config):
        parameters = self.set_parameters_train(parameters, config)
        before_loss, _,_ = self.evaluate(parameters=parameters,config=config)
        r = self.model.fit(self.x_train, self.y_train, epochs=self.epochs, validation_split=0.15, 
                        verbose=0,callbacks=self.get_callbacks(int(config['round'])))
        hist = r.history
        after_loss, _, _ = self.evaluate(parameters=self.model.get_weights(),config=config)
        return self.model.get_weights(), len(self.x_train), {'hidx': self.hidx, 'before': before_loss, 'after': after_loss}

    

    def get_callbacks(self,server_round : int):
        if self.save_train_res == True:
            self.callbacks.append(CSVLoggerWithLr(filename=self.file_path,append=True,server_round=server_round))
        if self.hpo == True:
            log(INFO,f"+++ Running With HPO +++ Round : {server_round}")
            self.callbacks = [es,reduce_lr,CSVLoggerWithLr(filename=self.file_path,append=True,server_round=server_round)]
        return self.callbacks
            

    
    def evaluate(self, parameters, config):
        try:
            self.model.set_weights(parameters)
        except ValueError:
            parameters = parameters[:-1]
            self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test, verbose=1)
        print("Eval accuracy on Client {} : {}".format(self.client_name,accuracy))
        return loss, len(self.x_test), {"accuracy": accuracy}
    


    def _sample_hyperparams(self):
        # obtain new learning rate for this batch
        distribution = tfp.distributions.Categorical(probs=tf.constant(self.distribution, dtype=tf.float32))
        hyp_idx = distribution.sample().numpy().item()
        hyp_config = self.hyperparameters[hyp_idx]
        return hyp_config, hyp_idx

