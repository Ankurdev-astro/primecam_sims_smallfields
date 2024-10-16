#!/bin/bash

# ============================
# MPI Processes Monitoring Script
# ============================

# Define variables
LOG_FILE="../../logs/mpi_memory_sim.log"
MPI_PROCESS_NAME="sim_data_primecam_mpi"  # Replace with your actual MPI process name
SAMPLE_INTERVAL=5  # Sampling interval in seconds

# Initialize the log file with a header
echo "timestamp,pid,pcpu,pmem,vsz,rss,start,time,command" > "$LOG_FILE"

echo "Monitoring MPI processes: $MPI_PROCESS_NAME"
echo "Logging data to $LOG_FILE every $SAMPLE_INTERVAL second(s)."

# Function to fetch and log process information
log_process_info() {
    # Get current timestamp
    timestamp=$(date +"%Y-%m-%d %H:%M:%S")

    # Fetch all PIDs matching the MPI process name
    pids=$(pgrep -f "$MPI_PROCESS_NAME")

    # Check if any MPI processes are running
    if [ -z "$pids" ]; then
        echo "[$timestamp] No MPI processes found matching '$MPI_PROCESS_NAME'."
        return
    fi

    # Iterate over each PID and log the required information
    for pid in $pids; do
        # Use ps to get process details
        # Fields: PID, %CPU, %MEM, VSZ, RSS, START, TIME, COMMAND
        read -r pid_val pcpu pmem vsz rss start time comm <<< $(ps -p "$pid" -o pid=,pcpu=,pmem=,vsz=,rss=,start=,time=,comm=)

        # Append the data to the log file in CSV format
        echo "$timestamp,$pid_val,$pcpu,$pmem,$vsz,$rss,$start,$time,$comm" >> "$LOG_FILE"
    done
}

# Infinite loop to continuously log data
while true; do
    log_process_info
    sleep "$SAMPLE_INTERVAL"
done

