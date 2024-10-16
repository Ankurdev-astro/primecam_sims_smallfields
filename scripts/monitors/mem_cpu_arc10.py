import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the log file
df = pd.read_csv('../../logs/mpi_memory_sim.log', parse_dates=['timestamp'])


# Convert timestamp to datetime if not already
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Example: Plot CPU usage for each PID
unique_pids = df['pid'].unique()
print(unique_pids)

# Find the index of the first NaN
nan_index = np.where(np.isnan(unique_pids))[0][0]

# Create a new array discarding everything before and including the first NaN
# Discard the first pid
unique = unique_pids[nan_index + 2:]

print(unique)

plt.figure(figsize=(12, 8))
for pid in unique:
    pid_data = df[df['pid'] == pid]
    plt.plot(pid_data['timestamp'], pid_data['rss']/1e6, label=f'PID {pid}')

plt.xlabel('Time')
plt.ylabel('RSS [in GB]')
plt.title('Resouce Usage Over Time for MPI Processes')
plt.legend()
plt.tight_layout()
plt.show()

# Similarly, you can plot memory usage by replacing 'pcpu' with 'pmem' or 'rss'

