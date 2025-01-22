import matplotlib.pyplot as plt
import time

# Initialize the figure
plt.ion()  # Turn on interactive mode
fig, ax = plt.subplots(figsize=(12, 8), facecolor='black')
ax.set_facecolor('dimgray')

# Store data for plotting
pids = []
vsz_usage = []
rss_usage = []

# Plotting loop
while True:
    try:
        # Read the log file
        with open('memory_test.log', 'r') as file:
            lines = file.readlines()

        # Parse the data
        pids.clear()
        vsz_usage.clear()
        rss_usage.clear()

        for line in lines:
            parts = line.strip().split(None, 7)  # Split into max 8 parts
            if len(parts) >= 8:
                try:
                    pid = int(parts[0])
                    # cpu = float(parts[1])  # %CPU (not used in this plot)
                    # mem = float(parts[2])  # %MEM (not used in this plot)
                    vsz = float(parts[3]) / (1024 * 1024)  # Convert VSZ from KB to GB
                    rss = float(parts[4]) / (1024 * 1024)  # Convert RSS from KB to GB
                    # start = parts[5]  # START (not used in this plot)
                    # time_spent = parts[6]  # TIME (not used in this plot)
                    # command = parts[7]  # COMMAND (not used in this plot)

                    pids.append(pid)
                    vsz_usage.append(vsz)
                    rss_usage.append(rss)
                except ValueError:
                    # Skip lines where conversion fails
                    print(f"Skipping line due to ValueError: {line.strip()}")
                    continue

        if pids:
            # Clear previous plot
            ax.clear()
            
            # Plot VSZ and RSS for each PID as grouped bars
            bar_width = 0.35
            indices = range(len(pids))

            # Plot VSZ bars
            vsz_bars = ax.bar([i - bar_width/2 for i in indices], vsz_usage, width=bar_width, color='blue', label='VSZ (GB)')
            # Plot RSS bars
            rss_bars = ax.bar([i + bar_width/2 for i in indices], rss_usage, width=bar_width, color='green', label='RSS (GB)')

            # Identify and mark the maximum value for each PID
            max_values = []
            for i in indices:
                max_val = max(vsz_usage[i], rss_usage[i])
                max_values.append(max_val)
                # # Determine the x position for the marker
                # if vsz_usage[i] >= rss_usage[i]:
                #     x_pos = i - bar_width/2
                # else:
                #     x_pos = i + bar_width/2
                # y_pos = max_val
                # Plot a red dot at the top of the higher bar
                # ax.plot(x_pos, y_pos, 'ro')  # 'ro' means red circle

            # Set dynamic y-axis limit
            current_max = max(max_values)
            ax.set_ylim(0, current_max * 1.1)  # Add 10% padding above the max value

            ax.set_xlabel('Process ID', color='g', fontsize=16)
            ax.set_ylabel('Memory Usage (GB)', color='g',fontsize=16)
            ax.set_title('Real-Time VSZ and RSS of Processes', color='g', fontsize=20)
            ax.set_xticks(indices)
            ax.set_xticklabels(pids, rotation=45, ha='right')
            ax.legend(loc='upper right', fontsize=14)
            # Set spine colors to white
            for spine in ax.spines.values():
                spine.set_color('white')

            # Set tick colors
            ax.tick_params(axis='both', colors='cyan')
            plt.tight_layout()

            plt.draw()
            plt.pause(2)  # Update the plot every 2 seconds
        else:
            # print("No data to plot.")
            pass

    except KeyboardInterrupt:
        print("Plotting interrupted by user.")
        break  # Exit the loop on Ctrl+C
    except Exception as e:
        print(f"An error occurred: {e}")
        break


plt.ioff()  # Turn off interactive mode
plt.show()
