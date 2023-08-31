import h5py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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

# ... (previous code)

# Create a list of predicted event labels (excluding Go Cue)
predicted_events = ['Predicted 1st Press', 'Predicted 2nd Press', 'Predicted 3rd Press', 'Predicted 4th Press', 'Predicted 5th Press']

# Calculate intervals between actual timepoints
intervals = np.diff(finger_timing)

# Calculate offsets for the predicted timepoints
offsets = np.random.uniform(0, 0.2, size=len(predicted_events) - 1)  # Generating random offsets for all predicted events except the last one

# Generate an additional random offset for the remaining time between the last actual event and the end of the trial
remaining_time = 4500 - finger_timing[-1]
offsets = np.concatenate((offsets, [np.random.uniform(0, 0.2) for _ in range(5 - len(predicted_events))]))

# Calculate the predicted event times and labels based on the intervals and offsets
predicted_event_times = finger_timing[1:] + offsets * intervals

# ... (rest of the code)

# Define event labels for the actual events
event_labels = ['Go Cue', '1st Press', '2nd Press', '3rd Press', '4th Press', '5th Press']

# Concatenate event times including the Go Cue (0) and finger timing data
event_times = np.concatenate([[0], finger_timing])

# Define distinct colors for vertical lines
line_colors = ['r', 'g', 'b', 'c', 'm', 'y']

# Loop over unique trial numbers and create separate plots
# Loop over unique trial numbers and create separate plots
for trial in unique_trials:
    plt.figure(figsize=(10, 6))

    # Filter data for the current trial
    trial_indices = np.where(trial_numbers == trial)[0]
    trial_time_milliseconds = time_milliseconds[trial_indices]
    trial_mean_alpha_power = mean_alpha_power[trial_indices]
    trial_mean_beta_power = mean_beta_power[trial_indices]
    trial_mean_gamma_power = mean_gamma_power[trial_indices]

    # Plot mean alpha power with diminished opacity
    plt.plot(trial_time_milliseconds, trial_mean_alpha_power, label='Alpha Power', alpha=0.5)

    # Plot mean beta power with diminished opacity
    plt.plot(trial_time_milliseconds, trial_mean_beta_power, label='Beta Power', alpha=0.5)

    # Plot mean gamma power with diminished opacity
    plt.plot(trial_time_milliseconds, trial_mean_gamma_power, label='Gamma Power', alpha=0.5)

    # Marking Actual Event Timestamps
    for i, point in enumerate(event_times):
        color = 'k' if event_labels[i] == 'Go Cue' else 'gray'  # Color for actual events
        linestyle = '-' if event_labels[i] == 'Go Cue' else '--'  # Line style for actual events
        plt.axvline(x=point, color=color, linestyle=linestyle, label=event_labels[i], alpha=0.8)

    # Marking Predicted Event Timestamps with 80% Accuracy
    for i, pred_time in enumerate(predicted_event_times):
        # Find the closest actual event timepoint to the predicted time
        closest_actual_time = event_times[np.argmin(np.abs(event_times - pred_time))]

        # Calculate the time difference between predicted and closest actual event time
        time_difference = np.abs(pred_time - closest_actual_time)

        # If the time difference is within 80% of the actual event interval, mark as predicted event
        if time_difference <= 0.8 * (event_times[1] - event_times[0]):
            color = 'r'
            linestyle = '-'
            label = f'{predicted_events[i]}'
            plt.axvline(x=pred_time, color=color, linestyle=linestyle, label=label, alpha=0.8)

    plt.xlabel('Time (ms)')
    plt.ylabel('Percent Power (%)')
    plt.title(f'Power of Alpha, Beta, and Gamma Oscillations (Trial {trial})')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.show()