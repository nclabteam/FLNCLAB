## A Simple Flower based framework for federated Learning.

### Note: This code is written and tested on Ubuntu and can work easily on any linux based distribution. For windows users some steps needs to changed.

### Steps to use this repo

1. Clone this repository using command
```bash
 git clone https://github.com/nclabteam/FLNCLAB.git
```
2. After clone cd into cloned directory and open terminal.

3. Ensure pip is installed if not install using
```bash
 sudo apt install python3-pip
```

4. We will be using miniconda to create a virtual environment, download miniconda as
If You already have conda installed skip step 4 and 5.

```bash
 curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o Miniconda3-latest-Linux-x86_64.sh
```
5. Then install miniconda using below command
```bash
  bash Miniconda3-latest-Linux-x86_64.sh
```
6. This framework is using Tensorflow and if you want to use GPU please refer to the official documentation of tensorflow on how to setup GPU.
at https://www.tensorflow.org/install/pip

7. Create a new virtual environment using conda
```bash
 conda env create -f environment.yml
```
It will create a virtual env named `venv-flwr` based on `environment.yml` file

8. For using virtual environment we need to activate the environment first.
```bash
 conda deactivate
 conda activate venv-flwr
```
8. We can change the confugration as per our need in config.yaml file

9. We can run server and clients using provided bash scripts using saperate terminals as :
```bash
 ./run_server.sh
```

```bash
 ./run_clients.sh
```
10. Or if we need to scale clients to few hundred we can run flower in simulation mode on single machine like:
```bash
python run_simulation.py
```
  This script will read the confugration from `config.yaml` file and starts the simulation.

  The outputs will be saved in `out` directory.


## Description about  [`config.yaml`](/config.yaml) file
The `config.yaml` file is a configuration file for this framework that trains a Federated Learning model.

The configuration file is divided into three sections: `common`, `server`, and `client`.

### Common Section
The `common` section contains the common configurations used in this framework. 

- `data_type` : This field specifies the data distribution type used in the training process. Currently supported data distributions are [ `iid`, `one-class-niid`, `one-class-niid-majority`, `two-class-niid`, or `dirichlet_niid` ] . Detailed explination can be found [here](./docs/data_distribution.md)
- `hpo` : This field specifies whether hyperparameter optimization should be used in the training process. It could be either `true` or `false`.
- `dataset` : This field specifies the dataset used in the training process. Currently supported data distributions are [ `fashionmnist`, `mnist`,`cifar-10` ]. Detailed explination can be found [here](./docs/datasets.md)
- `dirichlet_alpha` : This field is used when `data_type` is set to `dirichlet_niid`. It specifies the Dirichlet concentration parameter.
- `target_acc` : This field specifies the target accuracy that the model needs to achieve. It can take any value greater than `0`.
- `model` : This field specifies the model architecture used in the training process. Currently Implemented models are [  `mobilenetv2`, `simplecnn`, `simplednn`, `kerasexpcnn`, `mnistcnn`, `efficientnet`, `fedavgcnn`, `fmcnn` ]. Detailed explination can be found [here](./docs/models.md)
- `optimizer` : This field specifies the optimizer used in the training process. It could be either `sgd` or `adam`.
- `simulation` : This field specifies whether the training process is run as a simulation or not.

### Server Section
The `server` section contains the configurations for the server that coordinates the Federated Learning process.

- `max_rounds` : This field specifies the maximum number of rounds for the training process.
- `address` : This field specifies the IP address of the server.
- `fraction_fit` : This field specifies the fraction of participating clients used for training in each round.
- `min_fit_clients` : This field specifies the minimum number of participating clients required for training in each round.
- `fraction_evaluate` : This field specifies the fraction of participating clients used for evaluation in each round.
- `min_avalaible_clients` : This field specifies the minimum number of clients that should be available for the training process.
- `strategy` : This field specifies the strategy used for Federated Learning. Currently supported strategies are [ `fedavg`, `fedyogi`, `fedadagrad` ,`fedavgm` ] Detailed explination can be found [here](./docs/strategies.md)

### Client Section
The `client` section contains the configurations for the clients participating in the Federated Learning process.

- `epochs` : This field specifies the number of epochs for each client's training process.
- `batch_size` : This field specifies the batch size for each client's training process.
- `lr` : This field specifies the learning rate for each client's training process.
- `save_train_res` : This field specifies whether to save the training results. It could be either `true` or `false`.
If `save_train_res` is set to `true`, all the output data like accuracy, loss, time of each round would be saved in the `out` directory.
- `gpu` : True or False, Use GPU for training or not. Default to False
  `num_cpus` : 1  # no. of CPU cores that are assigned for each actor Default to 1
  `num_gpus` : Fraction of GPU assigned to each actor. (num_cpus and num_gpus can only used in simulation mode if `simulation` is set to `True`) For more details on this please refer to https://flower.dev/docs/framework/how-to-run-simulations.html and https://docs.ray.io/en/latest/ray-core/scheduling/resources.html

### Shakespeare
The `shakespeare` section is only needed if you need to evaluvate shakespeare dataset for next character prediction task.

- `sequence_length` : the length of input sequences characters
- `vocab_size ` : size of vocabulary
- `train_file` : path of processed shakespare train file
- `test_file` : path of processed shakespare test file

The shakespeare dataset can be downloaded by running 
```bash
 ./preprocessing/get_shakespeare_data.sh out_dir
```
 `out_dir` : (path where you want to save the data) and then copy the path of train and test file in the config.yaml file.
if you cannot run the bash file then run the below command in terminal
```bash
 chmod +x ./preprocessing/get_shakespeare_data.sh
```

