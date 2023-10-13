from mak.utils import generate_config_client, gen_out_file_client, generate_config_simulation
from mak.custom_clients.flwr_client import FlwrClient
from mak.custom_clients.fedex_client import FedExClient
from mak.data.fashion_mnist import FashionMnistData
from mak.data.mnist import MnistData
from mak.data.cifar_10_data import Cifar10Data
from mak.data.shakespeare import ShakespeareData
from mak.data.violation_detection import ViolationDetection
from mak.model.models import SimpleCNN, SimpleDNN, KerasExpCNN
import os
import flwr as fl

from mak.utils import  create_model, compile_model
from mak.utils import parse_args
# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

def generate_client(cid : str) -> fl.client.Client:
    client_config = generate_config_simulation(c_id=int(cid))
    data_type = client_config['data_type']
    lr = client_config['lr']
    total_clients = client_config['min_avalaible_clients']
    out_file_dir = gen_out_file_client(client_config)

    dataset = client_config['dataset']
    if dataset == 'cifar-10':
        input_shape = (32, 32, 3)
        num_classes = 10
    elif dataset == 'shakespeare':
        input_shape = (client_config['shakespeare']['sequence_length'])
        num_classes = (client_config['shakespeare']['vocab_size'])
    elif client_config['dataset'] == 'cifar-10':
        input_shape = (32, 32, 3)
        num_classes = 10
    elif dataset == 'violation-detection':
        input_shape = (client_config['violation-detection']['image_size'],client_config['violation-detection']['image_size'], 3)
        num_classes = client_config['violation-detection']['num_classes']
    else:
        input_shape = (28, 28, 1)
        num_classes = 10

    print(f"Dataset : {dataset}, Input Shape : {input_shape}, # Classes : {num_classes}")
    model = create_model(client_config['model'],input_shape=input_shape,num_classes=num_classes)
    compile_model(model,client_config['optimizer'],lr=lr)
    if dataset == 'mnist':
        data = MnistData(total_clients, data_type)
    elif dataset == 'cifar-10':
        data = Cifar10Data(total_clients, data_type)
    elif dataset == 'shakespeare':
        data = ShakespeareData(num_clients=total_clients,train_file=client_config['shakespeare']['train_file'],test_file=client_config['shakespeare']['test_file'])
    elif dataset == 'violation-detection':
        data = ViolationDetection(total_clients,client_config['violation-detection']['data_root'],image_size=input_shape[:2])
    else:
        data = FashionMnistData(total_clients, data_type)
    client_name = f"client_{cid}"

    # Load a subset of dataset to simulate the local data partition
    if dataset == 'shakespeare':
        (x_train, y_train), (x_test, y_test) = data.get_client_data(cid=client_config['partition'])
    elif dataset == 'violation-detection':
        (x_train, y_train), (x_test, y_test) = data.get_client_data(f"Client{cid}")
    else:
        print(f"Data Type : {client_config['data_type']}")
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

    print("Dataset : {}  Data Shape  : {}".format(dataset,x_train.shape))
    # Start Flower client
    
    if client_config['strategy'] == 'fedex':
        client =  FedExClient( config = client_config,
                              model = model,
                              xy_train = (x_train, y_train), 
                              xy_test = (x_test, y_test),
                              epochs=client_config['epochs'],
                              batch_size = client_config['batch_size'],
                              client_name = client_name,
                              hpo = client_config['hpo'],
                              file_path = out_file_dir,
                              save_train_res = client_config['save_train_res'],
                              )
    else:
        client = FlwrClient( model = model,
                            xy_train = (x_train, y_train), 
                            xy_test = (x_test, y_test),
                            epochs=client_config['epochs'],
                            batch_size = client_config['batch_size'],
                            hpo = client_config['hpo'],
                            client_name = client_name,
                            file_path = out_file_dir,
                            save_train_res = client_config['save_train_res'],
                            )
    return client


if __name__ == "__main__":
    args = parse_args()
    client_config = generate_config_client(args)
    client = generate_client(cid=client_config['client_id'])
    fl.client.start_numpy_client(
        server_address="127.0.0.1:8080",
        client=client,
    )
