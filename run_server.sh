#!/bin/bash

# python3 server.py --max_rounds="10" --min_fit_clients="2" --min_avalaible_clients="2" --hpo="1" --data_type="iid"&
python3.8 server.py --config="config.yaml" &
#sleep 5 # Sleep for 3s to give the server enough time to start

# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait
