#!/bin/bash

# Array of parameters
datasets=("three_state" "four_state" "eight_state")
dhs=(5 1)

echo "Starting all 6 experiments in parallel..."

for ds in "${datasets[@]}"; do
    for dh in "${dhs[@]}"; do
        echo "Launching: dataset=${ds}, dh=${dh}, delta=0.5"
        log_file="log_run_${ds}_dh${dh}.txt"
        uv run python examples/run_plots.py --dataset ${ds} --dh ${dh} --delta 0.5 --train > ${log_file} 2>&1 &
    done
done

echo "Waiting for all experiments to finish..."
wait
echo "All experiments finished!"
