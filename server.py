import argparse
import os
import csv
from datetime import date, datetime
from tabnanny import verbose
from mak.custom_strategy.custom_fedavg import CustomFedAvg
from mak.model.models import SimpleCNN, SimpleDNN, KerasExpCNN
from mak.custom_server import ServerSaveData
import tensorflow as tf
import flwr as fl

import yaml
from mak.utils import generate_config_server,gen_dir_outfile_server, get_strategy
from mak.utils import set_seed, create_model, compile_model,get_eval_fn, fit_config


def main() -> None:
    set_seed(13)
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument("--config",type=str,default = "config.yaml")
    args = parser.parse_args()
    server_config = generate_config_server(args)
    out_file_path = gen_dir_outfile_server(config=server_config)
    dataset = server_config['dataset']

    if server_config['dataset'] == 'cifar-10':
        input_shape = (32, 32, 3)
    else:
        input_shape = (28, 28, 1)

    model = create_model(server_config['model'],input_shape=input_shape,num_classes=10)
    # Compile model
    compile_model(model,server_config['optimizer'],server_config['lr'])
    num_clients = server_config['min_avalaible_clients']
    strategy = get_strategy(config=server_config,get_eval_fn=get_eval_fn,model=model,
                            dataset=dataset,num_clients=num_clients,on_fit_config_fn=fit_config)

    print(f"Using Strategy : {strategy.__class__}")

  # Start Flower server for four rounds of federated learning
    server = ServerSaveData(
        strategy=strategy, client_manager=fl.server.client_manager.SimpleClientManager(),out_file_path=out_file_path,target_acc=server_config['target_acc'])
    fl.server.start_server(
        server=server,
        server_address="[::]:8080",
        config=fl.server.ServerConfig(num_rounds=server_config['max_rounds']),
        strategy=strategy
    )


if __name__ == "__main__":
    main()
