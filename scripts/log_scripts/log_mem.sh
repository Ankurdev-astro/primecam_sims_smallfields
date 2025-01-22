#!/bin/bash

# Define the log file
LOG_FILE="memory_test.log"

# Infinite loop to log data every second
while true; do
    # Capture the process information and overwrite the log file
    ps -eo pid,pcpu,pmem,vsz,rss,start,time,command --sort=+pid | \
    grep make_ml_map_primecam | \
    # grep sim_data_primecam_mpi | \
    grep -v grep | grep -v sh | grep -v log_memory.sh | awk '{print $1, $2, $3, $4, $5, $6, $7, $8}' > "$LOG_FILE"
    
    # Sleep for 1 second before the next iteration
    sleep 1
done
