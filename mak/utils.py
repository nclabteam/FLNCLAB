import yaml
import os
from datetime import date, datetime
import csv
import json
import random
import numpy as np
import tensorflow as tf

def generate_config_server(args):
    yaml_file = args.config
    with open(file=yaml_file) as file:
        try:
            config = yaml.safe_load(file)   
            server_config = {}

            server_config['max_rounds'] = config['server']['max_rounds']
            server_config['min_fit_clients'] = config['server']['min_fit_clients']
            server_config['min_avalaible_clients'] = config['server']['min_avalaible_clients']
            server_config['strategy'] = config['server']['strategy']
            server_config['dataset']  = config['common']['dataset']
            server_config['epochs'] = config['client']['epochs']
            server_config['hpo'] = config['common']['hpo']
            server_config['data_type'] = config['common']['data_type']
            server_config['target_acc'] = config['common']['target_acc']
            server_config['dataset'] = config['common']['dataset']
            server_config['lr'] = config['client']['lr']
            server_config['batch_size'] = config['client']['batch_size']

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
            client_config['batch_size'] = config['client']['batch_size']
            client_config['hpo'] = config['common']['hpo']
            client_config['dataset'] = config['common']['dataset']
            client_config['data_type'] = config['common']['data_type']
            client_config['strategy'] = config['server']['strategy']
            client_config['server_address'] = config['server']['address']
            client_config['dataset'] = config['common']['dataset']
            client_config['lr'] = config['client']['lr']
            client_config['batch_size'] = config['client']['batch_size']

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
        json.dump(config, fp)

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
        today), mode, config['strategy'], data_dist_type, str(last_updated_dir))

    if not os.path.exists(final_dir_path):
        try:
            os.mkdir(final_dir_path)
        except FileExistsError:
            pass

    now = datetime.now()
    current_time = now.strftime("%H-%M-%S")

    file_path = os.path.join(final_dir_path,f"{current_time}_client_{config['client_id']}.csv")

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
