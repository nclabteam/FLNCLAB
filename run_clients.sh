#!/bin/bash

for i in `seq 0 1`; do
    p_id=$(( i % 10))
    echo "Starting client $i $p_id"
    # python3 client.py --partition=${p_id} --client_id=${i} --epochs="3" --hpo="1" --data_type="iid"  &
    python client.py --config="config.yaml" --partition=${p_id} --client_id=${i} &
done

# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait
