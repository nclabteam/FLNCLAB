import flwr as fl
import string
import csv
from typing import Callable, Dict, List, Optional, Tuple
from numproto import proto_to_ndarray, ndarray_to_proto
import pandas as pd
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    Parameters,
    Scalar,
)

from scipy.special import logsumexp
from numpy.linalg import norm
import numpy as np
from mak.fedex.hyperparameters import Hyperparameters


def discounted_mean(series, gamma=1.0):
    weight = gamma ** np.flip(np.arange(len(series)), axis=0)
    return np.inner(series, weight) / weight.sum()

class CustomFedEx(fl.server.strategy.FedAvg):
    def __init__(self,config,fraction_fit,fraction_eval,min_fit_clients,min_eval_clients,min_available_clients,eval_fn,initial_parameters,on_fit_config_fn = None) -> None:
        super().__init__(fraction_fit=fraction_fit,fraction_evaluate=fraction_eval,min_fit_clients=min_fit_clients,min_evaluate_clients=min_eval_clients,
                         min_available_clients=min_available_clients,evaluate_fn=eval_fn,initial_parameters=initial_parameters,on_fit_config_fn=on_fit_config_fn)
        self.config = config
        self.hyperparams = Hyperparameters(self.config['hyperparam_config_nr']) #hyperparmeter search space 
        self.hyperparams.save(self.config['hyperparam_file']) 
        self.fraction_fit = fraction_fit
        self.eval_fn = eval_fn
        self.initial_parameters = initial_parameters
        self.on_fit_config_fn = on_fit_config_fn
        self.log_distribution = np.full(len(self.hyperparams), -np.log(len(self.hyperparams)))
        self.distribution = np.exp(self.log_distribution)
        self.eta = np.sqrt(2*np.log(len(self.hyperparams)))
        self.discount_factor = 0.9,
        self.use_gain_avg = False
        self.distribution_history = []
        self.gain_history = [] # initialize with [0] to avoid nan-values in discounted mean
        self.log_gain_hist = []

        print("+++++++++++++++ Fedex intilized")

    def initialize_parameters(self, client_manager: fl.server.client_manager.ClientManager):
        """
        Initialize the model before training starts. Initialization sneds weights of initial_net
        passed in constructor

        Args:
            client_manager (fl.server.client_manager.ClientManager): Client Manager

        Returns:
            _type_: Initial model weights, distribution and hyperparameter configurations.
        """
        serialized_dist = ndarray_to_proto(self.distribution)
        self.initial_parameters.tensors.append(serialized_dist.ndarray)
        return self.initial_parameters

    def aggregate_fit(
        self,
        rnd,
        results,
        failures
    ):
        # obtain client weights
        samples = np.array([fit_res[1].num_examples for fit_res in results]) # a list os number of samples used by each client
        weights = samples / np.sum(samples) # for weighted
        aggregated_weights = super().aggregate_fit(rnd, results, failures)
        # log current distribution
        self.distribution_history.append(self.distribution)
        dh = np.array(self.distribution_history)
        df = pd.DataFrame(dh)
        df.to_csv('distribution_history.csv')
        rh = np.array(self.log_gain_hist)
        df = pd.DataFrame(rh)
        df.to_csv('gain_history.csv')

        gains = self.compute_gains(weights, results)
        self.update_distribution(gains, weights)
        
        # sample hyperparameters and append them to the parameters
        serialized_dist = ndarray_to_proto(self.distribution)
        aggregated_weights[0].tensors.append(serialized_dist.ndarray)
        return aggregated_weights

    def compute_gains(self, weights, results):
        """
        Computes the average gains/progress the model made during the last fit-call.
        Each client computes its validation loss before and after a backpop-step.
        The difference before - after is averaged and we compute (avg_before - avg_after) - gain_history.
        The gain_history is a discounted mean telling us how much gain we have obtained in the last
        rounds. If we obtain a better gain than in history, we will emphasize the corresponding
        hyperparameter-configurations in the distribution, if not these configurations get less
        likely to be sampled in the next round.

        Args:
            weights (_type_): Client weights
            results (_type_): Client results

        Returns:
            _type_: Gains
        """
        after_losses = [res.metrics['after'] for _, res in results]
        before_losses = [res.metrics['before'] for _, res in results]
        hidxs = [res.metrics['hidx'] for _, res in results]
        # compute (avg_before - avg_after)
        avg_gains = np.array([w * (a - b) for w, a, b in zip(weights, after_losses, before_losses)]).sum()
        self.gain_history.append(avg_gains)
        gains = []
        # use gain-history to obtain how much we improved on "average" in history
        baseline = discounted_mean(np.array(self.gain_history), self.discount_factor) if len(self.gain_history) > 0 else 0.0
        for hidx, al, bl, w in zip(hidxs, after_losses, before_losses, weights):
            gain = w * ((al - bl) - baseline) if self.use_gain_avg else w * (al - bl)
            client_gains = np.zeros(len(self.hyperparams))
            client_gains[hidx] = gain
            gains.append(client_gains)
        gains = np.array(gains)
        gains = gains.sum(axis=0)
        self.log_gain_hist.append(gains)
        return gains
    
    def update_distribution(self, gains, weights):
        """
        Update the distribution over the hyperparameter-search space.
        First, an exponantiated "gradient" update is made based on the gains we obtained.
        As a following step, we bound the maximum probability to be epsilon.
        Those configurations which have probability > epsilon after the exponantiated gradient step,
        are re-weighted such that near configurations are emphasized as well.
        NOTE: This re-weighting constraints our hyperparameter-search space to parameters on which an order can be defined.

        Args:
            gains (_type_): Gains obtained in last round
            weights (_type_): Weights of clients
        """
        denom = 1.0 if np.all(gains == 0.0) else norm(gains, float('inf'))
        self.log_distribution -= self.eta / denom * gains
        self.log_distribution -= logsumexp(self.log_distribution)
        self.distribution = np.exp(self.log_distribution)