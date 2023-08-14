from mak.utils import generate_config_client, gen_out_file_client, generate_config_simulation
from mak.custom_clients.flwr_client import FlwrClient
from mak.data.fashion_mnist import FashionMnistData
from mak.data.mnist import MnistData
from mak.data.cifar_10_data import Cifar10Data
from mak.model.models import SimpleCNN, SimpleDNN, KerasExpCNN
import os
import flwr as fl
import tensorflow as tf
import numpy as np
from typing import Dict, Tuple, cast
from mak.utils import set_seed, create_model, compile_model
from mak.utils import parse_args
# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def generate_client(cid : str) -> fl.client.Client:
    client_config = generate_config_simulation(c_id=int(cid))
    data_type = client_config['data_type']
    lr = client_config['lr']
    total_clients = client_config['min_avalaible_clients']
    out_file_dir = gen_out_file_client(client_config)
    if client_config['dataset'] == 'cifar-10':
        input_shape = (32, 32, 3)
    else:
        input_shape = (28, 28, 1)
    model = create_model(client_config['model'],input_shape=input_shape,num_classes=10)
    compile_model(model,client_config['optimizer'],lr=lr)
    if client_config['dataset'] == 'mnist':
        data = MnistData(total_clients, data_type)
    elif client_config['dataset'] == 'cifar-10':
        data = Cifar10Data(total_clients, data_type)
    else:
        data = FashionMnistData(total_clients, data_type)
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.categorical_crossentropy,
        metrics=["accuracy"],
        run_eagerly=True,
    )
    client_name = f"client_{cid}"
    print(f"Data Type : {client_config['data_type']}")
    # Load a subset of dataset to simulate the local data partition
    if data_type == "one-class-niid":
        print("Using One Class NoN-IID data")
        (x_train, y_train), (x_test, y_test) = data.load_data_one_class(
            class_id=client_config['partition'])
            
    elif data_type == "one-class-niid-majority":
        print("Using One Class NoN-IID data With Majority class = ",str(client_config['partition']))
        (x_train, y_train), (x_test, y_test) = data.load_data_majority_class(
            class_id=client_config['partition'],percent=0.75)

    elif data_type == "two-class-niid":
        class_1 = int(client_config['partition'])
        class_2 = int(class_1 % 9 + 1)
        print(f"Class 1 = {class_1}, Class 2 = {class_2}")
        (x_train, y_train), (x_test, y_test) = data.load_data_two_classes(
            class_1=class_1, class_2=class_2)
    elif data_type == "dirichlet-niid":
        alpha = client_config['dirichlet_alpha']
        print("Using Dirichlet Distribution with alpha = {}".format(client_config['dirichlet_alpha']))
        (x_train, y_train), (_, _) = data.load_data_niid_dirchlet(alpha=alpha,min_size=15,partition=client_config['client_id'])
        
        (x_test,y_test) = data.load_test_data()
    else:
        print("Using Default IID Settings")
        (x_train, y_train), (x_test, y_test) = data.load_data_iid(id=
            client_config['partition'])

    print("Data Shape  : {}".format(x_train.shape))
    # Start Flower client
    client = FlwrClient(model, (x_train, y_train), (x_test, y_test),
                                epochs=client_config['epochs'], batch_size=client_config['batch_size'],
                                  hpo=client_config['hpo'], client_name=client_name,file_path=out_file_dir,
                                  save_train_res = client_config['save_train_res'])
    return client


if __name__ == "__main__":
    args = parse_args()
    client_config = generate_config_client(args)
    client = generate_client(cid=client_config['client_id'])
    fl.client.start_numpy_client(
        server_address="127.0.0.1:8080",
        client=client,
    )
