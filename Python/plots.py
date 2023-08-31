import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from beh import data_dict

# Load the CSV file
csv_path = r'D:\MSc Project\Data\Old Data\MEG\s08\mean_power_data_python.csv'
data = pd.read_csv(csv_path, header=None)

# Assuming your CSV columns are in the order: Time, mean alpha power, mean beta power, mean gamma power
time_seconds = data.iloc[:, 0]
time_milliseconds = time_seconds * 1000.0  # Convert time to milliseconds
mean_alpha_power = data.iloc[:, 1]
mean_beta_power = data.iloc[:, 2]
mean_gamma_power = data.iloc[:, 3]

# Load finger timing data (replace this with your actual data)
finger_timing = data_dict['timing'][:, 2]

# Define event labels and their corresponding times
event_labels = ['Go Cue', '1st Press', '2nd Press', '3rd Press', '4th Press', '5th Press']
event_times = np.concatenate([[0], finger_timing])

# Define distinct colors for vertical lines
line_colors = ['r', 'g', 'b', 'c', 'm', 'y']

# Plotting the graph
plt.figure(figsize=(10, 6))

# Plot mean alpha power
plt.plot(time_milliseconds, mean_alpha_power, label='Alpha Power', marker='o')

# Plot mean beta power
plt.plot(time_milliseconds, mean_beta_power, label='Beta Power', marker='o')

# Plot mean gamma power
plt.plot(time_milliseconds, mean_gamma_power, label='Gamma Power', marker='o')

# Marking Timestamps
for i, point in enumerate(event_times):
    plt.axvline(x=point, color=line_colors[i], linestyle='--', label=event_labels[i])

plt.xlabel('Time (ms)')
plt.ylabel('Mean Power')
plt.title('Mean Power of Alpha, Beta, and Gamma Oscillations')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
