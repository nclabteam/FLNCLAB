import os
from mak.utils import get_strategy, generate_config_simulation, get_eval_fn, gen_dir_outfile_server
from mak.utils import save_simulation_history
from mak.utils import create_model, compile_model,get_eval_fn, fit_config
# Make TensorFlow logs less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import flwr as fl
from flwr.simulation.ray_transport.utils import enable_tf_gpu_growth
from client import generate_client
from mak.custom_server import ServerSaveData


def main() -> None:
    # Start Flower simulation
    config = generate_config_simulation(c_id=-1) # c_id = -1 implies the confugration is for server
    dataset = config['dataset']
    num_clients = config['min_avalaible_clients']
    if dataset == 'cifar-10':
        input_shape = (32, 32, 3)
        num_classes = 10
    elif dataset == 'shakespeare':
        input_shape = (config['shakespeare']['sequence_length'])
        num_classes = (config['shakespeare']['vocab_size'])
    else:
        input_shape = (28, 28, 1)
        num_classes = 10

    out_file_path = gen_dir_outfile_server(config=config)

    model = create_model(config['model'],input_shape=input_shape,num_classes=num_classes)
    # Compile model
    compile_model(model,config['optimizer'],config['lr'])
    strategy = get_strategy(config=config,get_eval_fn=get_eval_fn,model=model,dataset=dataset,num_clients=num_clients,on_fit_config_fn=fit_config)
    server = ServerSaveData(
        strategy=strategy, client_manager=fl.server.client_manager.SimpleClientManager(),out_file_path=out_file_path,target_acc=config['target_acc'])
    
    my_client_resources = {'num_cpus': 1, 'num_gpus': 0.05}

    hist = fl.simulation.start_simulation(
        client_fn=generate_client,
        num_clients=config['min_avalaible_clients'],
        config=fl.server.ServerConfig(num_rounds=config['max_rounds']),
        strategy = strategy,
        server = server,
        client_resources = my_client_resources,
        actor_kwargs={"on_actor_init_fn": enable_tf_gpu_growth},
    )
    
   
    simu_data_file_path = out_file_path.replace('.csv','_metrics.csv')
    save_simulation_history(hist=hist,path = simu_data_file_path)


if __name__ == "__main__":
    main()