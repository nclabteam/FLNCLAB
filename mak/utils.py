import yaml
import os
from datetime import date, datetime
import csv
import json
import random
import numpy as np
import tensorflow as tf
from mak.model.models import *
import argparse
import flwr as fl
from mak.data.fashion_mnist import FashionMnistData
from mak.data.mnist import MnistData
from mak.data.cifar_10_data import Cifar10Data
from mak.data.shakespeare import ShakespeareData
from mak.custom_strategy.fedex_strategy import CustomFedEx
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from flwr.common import Metrics
import json
import pandas as pd

def generate_config_server(args):
    yaml_file = args.config
    with open(file=yaml_file) as file:
        try:
            config = yaml.safe_load(file)   
            server_config = {}

            server_config['max_rounds'] = config['server']['max_rounds']
            server_config['fraction_fit'] = config['server']['fraction_fit']
            server_config['fraction_evaluate'] = config['server']['fraction_evaluate']
            server_config['min_fit_clients'] = config['server']['min_fit_clients']
            server_config['min_avalaible_clients'] = config['server']['min_avalaible_clients']
            server_config['strategy'] = config['server']['strategy']
            server_config['dataset']  = config['common']['dataset']
            server_config['epochs'] = config['client']['epochs']
            server_config['model'] = config['common']['model']
            server_config['hpo'] = config['common']['hpo']
            server_config['data_type'] = config['common']['data_type']
            server_config['target_acc'] = config['common']['target_acc']
            server_config['dataset'] = config['common']['dataset']
            server_config['lr'] = config['client']['lr']
            server_config['batch_size'] = config['client']['batch_size']
            server_config['optimizer'] = config['common']['optimizer']

            if config['fedex']:
                server_config['hyperparam_config_nr'] = config['fedex']['hyperparam_config_nr']
                server_config['hyperparam_file'] = config['fedex']['hyperparam_file']

            return server_config
        except yaml.YAMLError as exc:
            print(exc)


def generate_config_client(args):
    yaml_file = args.config
    with open(file=yaml_file) as file:
        try:
            config = yaml.safe_load(file)   
            client_config = {}
            client_config['partition'] = args.partition
            client_config['client_id'] = args.client_id
            client_config['epochs'] = config['client']['epochs']
            client_config['save_train_res'] = config['client']['save_train_res']
            client_config['batch_size'] = config['client']['batch_size']
            client_config['hpo'] = config['common']['hpo']
            client_config['dataset'] = config['common']['dataset']
            client_config['data_type'] = config['common']['data_type']
            client_config['dirichlet_alpha'] = config['common']['dirichlet_alpha']
            client_config['strategy'] = config['server']['strategy']
            client_config['server_address'] = config['server']['address']
            client_config['dataset'] = config['common']['dataset']
            client_config['lr'] = config['client']['lr']
            client_config['batch_size'] = config['client']['batch_size']
            client_config['model'] = config['common']['model']
            client_config['optimizer'] = config['common']['optimizer']
            client_config['min_avalaible_clients'] = config['server']['min_avalaible_clients']

            return client_config
        except yaml.YAMLError as exc:
            print(exc)

def find_file(dir,file_name):
    dirs = os.listdir(dir)
    dirs_sorted = sorted(dirs,reverse=True)
    for dir in dirs_sorted:
        fname,ext = os.path.splitext(dir)
        if ext == '.csv':
            f = ('_'.join(fname.split('_')[:-1]))
            if f == file_name:
                last_trail = fname.split('_')[-1]
                return last_trail
            else:
                return -1
    return -1

def gen_dir_outfile_server(config):
    # generates the basic directory structure for out data and the header for file
    today = date.today()
    mode = "hpo" if config['hpo'] == True else "nhpo"
    data_dist_type = config['data_type']
    BASE_DIR = "out"
    if not os.path.exists(BASE_DIR):
        os.mkdir(BASE_DIR)

    # create a date wise folder
    if not os.path.exists(os.path.join(BASE_DIR, str(today))):
        os.mkdir(os.path.join(BASE_DIR, str(today)))

    # create saperate folder based on hpo mode
    if not os.path.exists(os.path.join(BASE_DIR, str(today), mode)):
        os.mkdir(os.path.join(BASE_DIR, str(today), mode))

    # create saperate folder based on strategy
    if not os.path.exists(os.path.join(BASE_DIR, str(today), mode, config['strategy'])):
        os.mkdir(os.path.join(BASE_DIR, str(today), mode, config['strategy']))

    # create saperate folder based on data distribution type
    if not os.path.exists(os.path.join(BASE_DIR, str(today), mode, config['strategy'], data_dist_type)):
        os.mkdir(os.path.join(BASE_DIR, str(today),
                 mode, config['strategy'], data_dist_type))

    dirs = os.listdir(os.path.join(BASE_DIR, str(today),
                      mode, config['strategy'], data_dist_type))
    final_dir_path = os.path.join(BASE_DIR, str(
        today), mode, config['strategy'], data_dist_type, str(len(dirs)))

    if not os.path.exists(final_dir_path):
        os.mkdir(final_dir_path)

    now = datetime.now()
    current_time = now.strftime("%H-%M-%S")
    #save all confugration file as json file
    json_file_name = f"config.json"
    with open(os.path.join(final_dir_path,json_file_name), 'w') as fp:
        json.dump(config, fp,indent=4)

    file_name = f"{config['strategy']}_{config['dataset']}_{config['data_type']}_{config['batch_size']}_{config['lr']}_{config['epochs']}_{mode}"
    file_name = f"{file_name}.csv"
    out_file_path = os.path.join(
        final_dir_path, file_name)
    # create empty server history file
    if not os.path.exists(out_file_path):
        with open(out_file_path, 'w', encoding='UTF8') as f:
            # create the csv writer
            header = ["round", "accuracy", "loss", "time"]
            writer = csv.writer(f)
            writer.writerow(header)
            f.close()
    return out_file_path


def gen_out_file_client(config):
    # generates the basic directory structure for out data and the header for file
    today = date.today()
    mode = "hpo" if config['hpo'] == True else "nhpo"
    data_dist_type = config['data_type']
    BASE_DIR = "out"
    if not os.path.exists(BASE_DIR):
        try:
            os.mkdir(BASE_DIR)
        except FileExistsError:
            pass
    # create a date wise folder
    if not os.path.exists(os.path.join(BASE_DIR, str(today))):
        try:
            os.mkdir(os.path.join(BASE_DIR, str(today)))
        except FileExistsError:
            pass

    # create saperate folder based on hpo mode
    if not os.path.exists(os.path.join(BASE_DIR, str(today), mode)):
        try:
            os.mkdir(os.path.join(BASE_DIR, str(today), mode))
        except FileExistsError:
            pass

    # create saperate folder based on strategy
    if not os.path.exists(os.path.join(BASE_DIR, str(today), mode, config['strategy'])):
        try:
            os.mkdir(os.path.join(BASE_DIR, str(today), mode, config['strategy']))
        except FileExistsError:
            pass

    # create saperate folder based on data distribution type
    if not os.path.exists(os.path.join(BASE_DIR, str(today), mode, config['strategy'], data_dist_type)):
        try:
            os.mkdir(os.path.join(BASE_DIR, str(today),
                 mode, config['strategy'], data_dist_type))
        except FileExistsError:
            pass

    dirs = os.listdir(os.path.join(BASE_DIR, str(today),
                      mode, config['strategy'], data_dist_type))
    dirs_sorted = sorted(dirs)
    if len(dirs_sorted) >0:
        last_updated_dir = dirs[-1]
    else:
        last_updated_dir ='0'
    final_dir_path = os.path.join(BASE_DIR, str(
        today), mode, config['strategy'], data_dist_type, str(last_updated_dir),'clients')

    if not os.path.exists(final_dir_path):
        try:
            os.mkdir(final_dir_path)
        except FileExistsError:
            pass

    now = datetime.now()
    current_time = now.strftime("%H-%M-%S")

    file_path = os.path.join(final_dir_path,f"client_{config['client_id']}.csv")

    return file_path


def set_seed(seed: int = 13) -> None:
  random.seed(seed)
  np.random.seed(seed)
  tf.random.set_seed(seed)
  tf.experimental.numpy.random.seed(seed)
  # When running on the CuDNN backend, two further options must be set
  os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
  os.environ['TF_DETERMINISTIC_OPS'] = '1'
  # Set a fixed value for the hash seed
  os.environ["PYTHONHASHSEED"] = str(seed)
  print(f"Random seed set as {seed}")


def create_model(name,input_shape, num_classes=10):
    # mobilenetv2, simplecnn, simplednn, kerasexpcnn, mnistcnn,efficientnet, lstm-shakespeare
    if name == 'mobilenetv2':
        return MobileNetV2(input_shape=input_shape,num_classes=num_classes)._model
    elif name == 'simplecnn':
        return SimpleCNN(input_shape=input_shape,num_classes=num_classes)._model
    elif name == 'simplednn':
        return SimpleDNN(input_shape=input_shape,num_classes=num_classes)._model
    elif name == 'kerasexpcnn':
        return KerasExpCNN(input_shape=input_shape,num_classes=num_classes)._model
    elif name == 'mnistcnn':
        return MNISTCNN(input_shape=input_shape,num_classes=num_classes)._model
    elif name == 'efficientnet':
        return EfficientNetB0(input_shape=input_shape,num_classes=num_classes)._model
    elif name == 'fedavgcnn':
        return FedAVGCNN(input_shape=input_shape,num_classes=num_classes)._model
    elif name == 'fmcnn':
        return FMCNNModel(input_shape=input_shape,num_classes=num_classes)._model
    elif name == 'resnet-18':
        return ResNet18(input_shape=input_shape,num_classes=num_classes)._model
    elif name == 'lstm-shakespeare':
        return LSTMModel(input_shape=input_shape,num_classes=num_classes)._model
    else:
        print("Invalid model name. Model name must be among [ mobilenetv2, simplecnn, simplednn, kerasexpcnn, mnistcnn,efficientnet,lstm-shakespeare, resnet-18]")

def compile_model(model,optimizer,lr= 0.001):
    if optimizer == 'sgd':
        opt = tf.keras.optimizers.SGD(learning_rate = lr)
    else:
        opt = tf.keras.optimizers.Adam(learning_rate = lr)
    
    model.compile(
        optimizer=opt,
        loss=tf.keras.losses.categorical_crossentropy,
        metrics=["accuracy"],
        run_eagerly=True,
    )

def generate_config_simulation(c_id):
    """
    Generates config for running the simulation based on `config.yaml`

    Parameters:
    c_id (int): client id of the simulated client
    """
    yaml_file = "config.yaml"
    with open(file=yaml_file) as file:
        try:
            config = yaml.safe_load(file)   
            simu_config = {}
            simu_config['partition'] = c_id
            simu_config['client_id'] = c_id
            simu_config['epochs'] = config['client']['epochs']
            simu_config['save_train_res'] = config['client']['save_train_res']
            simu_config['batch_size'] = config['client']['batch_size']
            simu_config['hpo'] = config['common']['hpo']
            simu_config['dataset'] = config['common']['dataset']
            simu_config['data_type'] = config['common']['data_type']
            simu_config['dirichlet_alpha'] = config['common']['dirichlet_alpha']
            simu_config['strategy'] = config['server']['strategy']
            simu_config['server_address'] = config['server']['address']
            simu_config['lr'] = config['client']['lr']
            simu_config['model'] = config['common']['model']
            simu_config['optimizer'] = config['common']['optimizer']
            simu_config['simulation'] = config['common']['simulation']
            simu_config['min_avalaible_clients'] = config['server']['min_avalaible_clients']
            simu_config['max_rounds'] = config['server']['max_rounds']
            simu_config['fraction_fit'] = config['server']['fraction_fit']
            simu_config['fraction_evaluate'] = config['server']['fraction_evaluate']
            simu_config['min_fit_clients'] = config['server']['min_fit_clients']
            simu_config['target_acc'] = config['common']['target_acc']

            if config['fedex']:
                simu_config['hyperparam_config_nr'] = config['fedex']['hyperparam_config_nr']
                simu_config['hyperparam_file'] = config['fedex']['hyperparam_file']

            if config['shakespeare']:
                simu_config['shakespeare'] ={}
                simu_config['shakespeare']['sequence_length'] = config['shakespeare']['sequence_length']
                simu_config['shakespeare']['vocab_size'] = config['shakespeare']['vocab_size']
                simu_config['shakespeare']['train_file'] = config['shakespeare']['train_file']
                simu_config['shakespeare']['test_file'] = config['shakespeare']['test_file']

            return simu_config
        except yaml.YAMLError as exc:
            print(exc)

def get_strategy(config,get_eval_fn,model,dataset,num_clients,on_fit_config_fn):
     # Create strategy
    if config['strategy'] == "fedyogi":
        strategy = fl.server.strategy.FedYogi(
            fraction_fit=config['fraction_fit'],
            fraction_evaluate= config['fraction_evaluate'],
            min_fit_clients=config['min_fit_clients'],
            min_evaluate_clients=2,
            min_available_clients=config['min_avalaible_clients'],
            evaluate_fn=get_eval_fn(model,dataset,num_clients,config),
            evaluate_metrics_aggregation_fn=agg_metrics,
            on_fit_config_fn=on_fit_config_fn,
            initial_parameters=fl.common.ndarrays_to_parameters(
                model.get_weights()),
            eta=1e-2,
            eta_l=0.0316,
            beta_1=0.9,
            beta_2=0.99,
            tau=1e-3,
        )
    elif config['strategy'] == "fedadagrad":
        strategy = fl.server.strategy.FedAdagrad(
            fraction_fit=config['fraction_fit'],
            fraction_evaluate= config['fraction_evaluate'],
            min_fit_clients=config['min_fit_clients'],
            min_evaluate_clients=2,
            min_available_clients=config['min_avalaible_clients'],
            evaluate_fn=get_eval_fn(model,dataset,num_clients,config),
            evaluate_metrics_aggregation_fn=agg_metrics,
            on_fit_config_fn=on_fit_config_fn,
            initial_parameters=fl.common.ndarrays_to_parameters(
                model.get_weights()),
            eta=1e-2,
            eta_l=0.0316,
            tau=1e-3,
        )
    elif config['strategy'] == "fedavgm":
        strategy = fl.server.strategy.FedAvgM(
            fraction_fit=config['fraction_fit'],
            fraction_evaluate= config['fraction_evaluate'],
            min_fit_clients=config['min_fit_clients'],
            min_evaluate_clients=2,
            min_available_clients=config['min_avalaible_clients'],
            evaluate_fn=get_eval_fn(model,dataset,num_clients,config),
            evaluate_metrics_aggregation_fn=agg_metrics,
            on_fit_config_fn=on_fit_config_fn,
            initial_parameters=fl.common.ndarrays_to_parameters(
                model.get_weights()),
            server_learning_rate=1.0,
            server_momentum=0.2,
        )
    elif config['strategy'] == "fedprox": #from flwr 1.XX
        strategy = fl.server.strategy.FedProx(
            fraction_fit=config['fraction_fit'],
            fraction_evaluate= config['fraction_evaluate'],
            min_fit_clients=config['min_fit_clients'],
            min_evaluate_clients=2,
            min_available_clients=config['min_avalaible_clients'],
            evaluate_fn=get_eval_fn(model,dataset,num_clients,config),
            evaluate_metrics_aggregation_fn=agg_metrics,
            on_fit_config_fn=on_fit_config_fn,
            proximal_mu = 0.5,
            initial_parameters=fl.common.ndarrays_to_parameters(
                model.get_weights()),
        )
    elif config['strategy'] == "fedex":
        strategy = CustomFedEx(
            config=config,
            fraction_fit=config['fraction_fit'],
            eval_fn=get_eval_fn(model,dataset,num_clients,config),
            fraction_eval= config['fraction_evaluate'],
            min_fit_clients=config['min_fit_clients'],
            min_eval_clients=2,
            min_available_clients=config['min_avalaible_clients'],
            on_fit_config_fn=on_fit_config_fn,
            initial_parameters=fl.common.ndarrays_to_parameters(
                model.get_weights()),
        )   
    else:
        strategy = fl.server.strategy.FedAvg(
            evaluate_fn=get_eval_fn(model,dataset,num_clients,config),
            fraction_fit=config['fraction_fit'],
            fraction_evaluate= config['fraction_evaluate'],
            min_fit_clients=config['min_fit_clients'],
            min_evaluate_clients=2,
            min_available_clients=config['min_avalaible_clients'],
            evaluate_metrics_aggregation_fn=agg_metrics,
            on_fit_config_fn=on_fit_config_fn,
            initial_parameters=fl.common.ndarrays_to_parameters(
                model.get_weights()),
        )
    return strategy

def agg_metrics(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}

def get_eval_fn(model,dataset,num_clients,config):
    """Return an evaluation function for server-side evaluation."""
    # Load data and model here to avoid the overhead of doing it in `evaluate` itself
    if dataset =='mnist':
        (x_val, y_val) = MnistData(num_clients=num_clients).load_test_data()
    elif dataset == 'cifar-10':
        (x_val, y_val) = Cifar10Data(num_clients=num_clients).load_test_data()
    elif dataset == 'shakespeare':
        input_shape = (config['shakespeare']['sequence_length'])
        num_classes = (config['shakespeare']['vocab_size'])
        (x_val,y_val) = ShakespeareData(num_clients=num_clients,train_file=config['shakespeare']['train_file'],test_file=config['shakespeare']['test_file']).load_test_data()
    else:
        (x_val, y_val) = FashionMnistData(num_clients=num_clients).load_test_data()
    print("Validation x shape : {}".format(x_val.shape))

    # The `evaluate` function will be called after every round
    def evaluate(
        server_round: int,
        parameters: fl.common.NDArrays,
        config: Dict[str, fl.common.Scalar],
    ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
        try:
            model.set_weights(parameters)
        except ValueError:
            parameters = parameters[:-1]
            model.set_weights(parameters)
        loss, accuracy = model.evaluate(x_val, y_val)
        model.save('./saved_model')
        print("Accuracy {} ".format(accuracy))
        return loss, {"accuracy": accuracy}

    return evaluate
def parse_args():
    set_seed(13)
     # Parse command line argument `partition`
    parser = argparse.ArgumentParser(description="Federated Hyper-parameter Optimisation")
    parser.add_argument("--partition", type=int, default=-1)
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--client_id", type=int, default=-1)
    args,unknown = parser.parse_known_args()
    return args

def save_simulation_history(hist : fl.server.history.History, path):
    losses_distributed = hist.losses_distributed
    losses_centralized = hist.losses_centralized
    metrics_distributed_fit = hist.metrics_distributed_fit
    metrics_distributed = hist.metrics_distributed
    metrics_centralized = hist.metrics_centralized

    rounds = []
    losses_centralized_dict = {}
    losses_distributed_dict = {}
    accuracy_distributed_dict = {}
    accuracy_centralized_dict = {}

    for loss in losses_centralized:
        c_rnd = loss[0]
        rounds.append(c_rnd)
        losses_centralized_dict[c_rnd] = loss[1]

    for loss in losses_distributed:
        c_rnd = loss[0]
        losses_distributed_dict[c_rnd] = loss[1]
    if 'accuracy' in metrics_distributed.keys():
        for acc in metrics_distributed['accuracy']:
            c_rnd = acc[0]
            accuracy_distributed_dict[c_rnd] = acc[1]
    if 'accuracy' in metrics_centralized.keys():
        for acc in metrics_centralized['accuracy']:
            c_rnd = acc[0]
            accuracy_centralized_dict[c_rnd] = acc[1]

    if len(metrics_distributed_fit) != 0:
        pass # TODO  check its implemetation later

    data = {"rounds" :rounds, "losses_centralized":losses_centralized_dict,"losses_distributed":losses_distributed_dict,
                 "accuracy_distributed": accuracy_distributed_dict,"accuracy_centralized" :accuracy_centralized_dict}
    
    # Create an empty DataFrame
    df = pd.DataFrame()

    # Iterate over each key in the data dictionary
    for key in data.keys():
        # If the key is 'rounds', set the 'rounds' column of the DataFrame to the rounds list
        if key == 'rounds':
            df['rounds'] = data[key]
        # Otherwise, create a new column in the DataFrame with the key as the column name
        else:
            column_data = []
            # Iterate over each round in the 'rounds' list and add the corresponding value for the current key
            for round_num in data['rounds']:
                # If the round number does not exist in the current key's dictionary, set the value to None
                if round_num not in data[key]:
                    column_data.append(None)
                else:
                    column_data.append(data[key][round_num])
            df[key] = column_data
    df.to_csv(path)

    json_obj = json.dumps(data,indent=4)
    with open(path.replace('.csv','.json'),"w") as outfile:
        outfile.write(json_obj)

def fit_config(server_round: int):
    """Return training configuration dict for each round.

    passes the current round number to the client
    """
    config = {
        "round": server_round,
    }
    return config

