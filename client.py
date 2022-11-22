from mak.utils import generate_config_client, gen_out_file_client
from mak.custom_clients.fashion_mnist_client import FashionMnistClient
from mak.data.fashion_mnist import FashionMnistData
from mak.data.mnist import MnistData
from mak.model.models import SimpleCNN, SimpleDNN, KerasExpCNN
import os
import argparse
import string
import flwr as fl
import tensorflow as tf
import numpy as np
from flwr.common import weights_to_parameters
from typing import Dict, Tuple, cast
from mak.utils import set_seed, create_model, compile_model
# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


# np.random.seed(0)
input_shape = (28, 28, 1)


def main() -> None:
    set_seed(13)
    # Parse command line argument `partition`
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument("--partition", type=int,
                        choices=range(0, 10), required=True)
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--client_id", type=int, required=True)
    args = parser.parse_args()
    client_config = generate_config_client(args)
    data_type = client_config['data_type']
    lr = client_config['lr']
    out_file_dir = gen_out_file_client(client_config)
    model = create_model(client_config['model'],input_shape=input_shape,num_classes=10)
    compile_model(model,client_config['optimizer'],lr=lr)
    if client_config['dataset'] == 'mnist':
        data = MnistData(10, data_type)
    else:
        data = FashionMnistData(10, data_type)
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.categorical_crossentropy,
        metrics=["accuracy"],
        run_eagerly=True,
    )

    client_name = f"client_{client_config['client_id']}"
    print(f"Data Type : {client_config['data_type']}")
    # Load a subset of fashion mnist to simulate the local data partition
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
    else:
        print("Using Default IID Settings")
        (x_train, y_train), (x_test, y_test) = data.load_data_iid(id=
            client_config['partition'])

    print("Data Shape  : {}".format(x_train.shape))
    # Start Flower client
    client = FashionMnistClient(model, (x_train, y_train), (x_test, y_test),
                                epochs=client_config['epochs'], batch_size=client_config['batch_size'], hpo=client_config['hpo'], client_name=client_name,file_path=out_file_dir)

    fl.client.start_numpy_client(f"{client_config['server_address']}:8080", client=client)


if __name__ == "__main__":
    main()
