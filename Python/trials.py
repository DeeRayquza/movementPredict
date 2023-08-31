import h5py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial.distance import cdist

# Load the MATLAB .mat file
path = r'D:\MSc Project\Data\Old Data\Behavioural\s25\behDataMEG.mat'
mat_data = h5py.File(path)

# Function to convert HDF5 dataset to numpy array
def hdf5_to_numpy(hdf5_dataset):
    if isinstance(hdf5_dataset, h5py.Dataset):
        return np.array(hdf5_dataset)
    elif isinstance(hdf5_dataset, h5py.Group):
        return {name: hdf5_to_numpy(item) for name, item in hdf5_dataset.items()}

# Convert the entire data to Python data structures
data_dict = hdf5_to_numpy(mat_data['A'])

# Load the CSV file
csv_path = r'D:\MSc Project\Data\Old Data\MEG\s25\mean_power_data_python_bc.csv'
data = pd.read_csv(csv_path, header=None)

# Assuming your CSV columns are in the order: Time, mean alpha power, mean beta power, mean gamma power, trial number
time_seconds = data.iloc[:, 0]
time_milliseconds = time_seconds * 1000.0  # Convert time to milliseconds
mean_alpha_power = data.iloc[:, 1]
mean_beta_power = data.iloc[:, 2]
mean_gamma_power = data.iloc[:, 3]
trial_numbers = data.iloc[:, 4]  # Extract trial numbers

unique_trials = np.unique(trial_numbers)  # Get unique trial numbers

# Load finger timing data (replace this with your actual data)
finger_timing = data_dict['timing'][:, 2]

# Define event labels and their corresponding times
event_labels = ['Go Cue', '1st Press', '2nd Press', '3rd Press', '4th Press', '5th Press']
event_times = np.concatenate([[0], finger_timing])

# Define distinct colors for vertical lines
line_colors = ['r', 'g', 'b', 'c', 'm', 'y']

# Loop over unique trial numbers and create separate plots
for trial in unique_trials:
    plt.figure(figsize=(10, 6))

    # Filter data for the current trial
    trial_indices = np.where(trial_numbers == trial)[0]
    trial_time_milliseconds = time_milliseconds[trial_indices]
    trial_mean_alpha_power = mean_alpha_power[trial_indices]
    trial_mean_beta_power = mean_beta_power[trial_indices]
    trial_mean_gamma_power = mean_gamma_power[trial_indices]

    # Plot mean alpha power
    plt.plot(trial_time_milliseconds, trial_mean_alpha_power, label='Alpha Power')

    # Plot mean beta power
    plt.plot(trial_time_milliseconds, trial_mean_beta_power, label='Beta Power')

    # Plot mean gamma power
    plt.plot(trial_time_milliseconds, trial_mean_gamma_power, label='Gamma Power')

    # Marking Timestamps
    for i, point in enumerate(event_times):
        plt.axvline(x=point, color=line_colors[i], linestyle='--', label=event_labels[i])

    plt.xlabel('Time (ms)')
    plt.ylabel('Percent Power (%)')
    plt.title(f'Power of Alpha, Beta, and Gamma Oscillations (Trial {trial})')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.show()