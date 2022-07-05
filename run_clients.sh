#!/bin/bash

# echo "Starting server"
# python3 server.py &
# sleep 5 # Sleep for 3s to give the server enough time to start

 python3 client.py --config="config.yaml" --partition=0 --client_id=0 &

# for i in `seq 0 5`; do
#     p_id=$(( i % 10))
#     echo "Starting client $i $p_id"
#     # python3 client.py --partition=${p_id} --client_id=${i} --epochs="3" --hpo="1" --data_type="iid"  &
#     python3 client.py --config="config.yaml" --partition=${p_id} --client_id=${i} &
# done

# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait
