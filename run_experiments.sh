#!/bin/bash

# Array of parameters
datasets=("three_state" "four_state" "eight_state")
dhs=(5 1)

echo "Starting experiments (running max 2 in parallel to prevent memory/CPU overload)..."

N=2
jobs_running=0

for ds in "${datasets[@]}"; do
    for dh in "${dhs[@]}"; do
        echo "Launching: dataset=${ds}, dh=${dh}, delta=0.5"
        log_file="log_run_${ds}_dh${dh}.txt"
        uv run python examples/run_plots.py --dataset ${ds} --dh ${dh} --delta 0.5 --train > ${log_file} 2>&1 &
        
        ((jobs_running++))
        if [ "$jobs_running" -ge "$N" ]; then
            wait
            jobs_running=0
        fi
    done
done

echo "Waiting for all experiments to finish..."
wait
echo "All experiments finished!"
