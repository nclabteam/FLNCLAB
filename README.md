## Hyperparameter Optimisation in Flower Framework.

### Aim :- Use Flower for HPO without touching the core code of Flower.

#### Target:- Use custom strategies to accomplish the aim.

### Using venv

### Steps

1. Clone this repository using command
```bash
~$ git clone https://github.com/kundroomajid/flwr_v3.git
```
2. After clone cd into cloned directory and open terminal.

3. Ensure pip is installed if not install using
```bash
~$ sudo apt install python3-pip
```

4. We will be using venv to create a virtual environment, install venv using this command

```bash
~$ sudo apt install python3.8-venv
```

Where python3.8 implies your version of installed python. If you are using any other version of python change your version accordingly.

5. Create a new virtual environment
```bash
~$ python3 -m venv venv-flwr
```
Where "venv-flwr" is the name of the virtual environment that we want to create. We can choose any name that we like instead of venv-flwr.


After this step a new directory (venv-flwr) will be created.

6. For using virtual environment we need to activate the environment first.
```bash
~$ source venv-flwr/bin/activate
```
7. Now we can install the project requirements or dependencies inside virtual environment using terminal as:
```bash
~$ pip install -r requirements.txt
```
All the project dependencies will be installed.

8. We can change the confugration as per our need in config.yaml file

9. We can run server and clients using provided bash scripts as :
```bash
~$ ./run_server.sh
```

```bash
~$ ./run_clients.sh
```
