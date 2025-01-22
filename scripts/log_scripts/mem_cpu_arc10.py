import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# ============================
# MPI Processes Data Visualization
# ============================

# Load the log file
log_file_path = '../../logs/mpi_memory_sim.log'  # Update this path if necessary
df = pd.read_csv(log_file_path, parse_dates=['timestamp'])

# Convert timestamp to datetime if not already
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Extract unique PIDs
unique_pids = df['pid'].unique()
print("All unique PIDs:", unique_pids)

# Handle potential NaN values in unique_pids
# Find the index of the first NaN (if any)
if np.isnan(unique_pids).any():
    nan_index = np.where(np.isnan(unique_pids))[0][0]
    # Discard everything before and including the first NaN
    # Also, discard the first PID after NaN as per original script
    unique = unique_pids[nan_index + 2:]
else:
    unique = unique_pids  # No NaN values present

print("Filtered PIDs for plotting:", unique)

# Initialize the plot with two subplots: one for PCpu and one for RSS
fig, (ax_cpu, ax_mem) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)

# Initialize lists to hold plot handles and labels for the legend
handles = []
labels = []

# Plot CPU usage for each PID
for pid in unique:
    pid_data = df[df['pid'] == pid]
    # Plot PCpu on the first subplot
    line_cpu, = ax_cpu.plot(pid_data['timestamp'], pid_data['pcpu'], label=f'PID {pid}')
    # Plot RSS (converted to GB) on the second subplot
    line_mem, = ax_mem.plot(pid_data['timestamp'], pid_data['rss'] / 1e6, label=f'PID {pid}')
    # Collect handles and labels for the legend
    handles.append(line_cpu)
    labels.append(f'PID {pid}')

# Customize the CPU subplot
ax_cpu.set_ylabel('CPU Usage (%)')
ax_cpu.set_title('CPU Usage Over Time for MPI Processes')
ax_cpu.grid(True)

# Customize the Memory subplot
ax_mem.set_xlabel('Time')
ax_mem.set_ylabel('RSS [GB]')
ax_mem.set_title('Memory Usage Over Time for MPI Processes')
ax_mem.grid(True)

# Create a single legend for both subplots
# Place the legend outside the subplots to avoid overlapping
fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.95, 0.95))

# Adjust layout to make room for the legend
plt.tight_layout(rect=[0, 0, 0.95, 0.95])

# Optionally, improve the x-axis date formatting for better readability
import matplotlib.dates as mdates

# Define the date format for the x-axis
date_format = mdates.DateFormatter('%H:%M:%S')
ax_mem.xaxis.set_major_formatter(date_format)

# Rotate x-axis labels for better readability
plt.setp(ax_mem.xaxis.get_majorticklabels(), rotation=45, ha='right')

plt.suptitle("Resource Usage for 1 schedule with 400 detectors")

fname = 'mpi_resource_usage_d400.png'
#plt.savefig(os.path.join("../../logs/",fname), dpi=300, bbox_inches='tight')
plt.show()

