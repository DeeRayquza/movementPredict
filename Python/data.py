import h5py
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

# Load the MATLAB .mat file
path = r'D:\MSc Project\Data\Old Data\Behavioural\s08\behDataMEG.mat'
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
csv_path = r'D:\MSc Project\Data\Old Data\MEG\s08\mean_power_data_python.csv'
data = pd.read_csv(csv_path, header=None)

# Assuming your CSV columns are in the order: Time, mean alpha power, mean beta power, mean gamma power, trial number
time_seconds = data.iloc[:, 0]
time_milliseconds = time_seconds * 1000.0  # Convert time to milliseconds
trial_numbers = data.iloc[:, 4]  # Extract trial numbers

unique_trials = np.unique(trial_numbers)  # Get unique trial numbers

# Define event labels and their corresponding times
event_labels = ['Go Cue', '1st Press', '2nd Press', '3rd Press', '4th Press', '5th Press']

# Create an empty list to hold the event data
event_data = []

# Loop over unique trial numbers
for trial in unique_trials:
    # Filter data for the current trial
    trial_indices = np.where(trial_numbers == trial)[0]
    trial_time_milliseconds = time_milliseconds[trial_indices].to_numpy()  # Convert to NumPy array

    # Initialize trial_event_labels with zeros
    trial_event_labels = ['0'] * len(trial_time_milliseconds)

    # Load finger timing data for the current trial
    finger_timing = data_dict['timing'][:, trial - 1]  # Subtract 1 because trial numbers are 1-based

    # Combine event times with trial-specific finger timing
    event_times = np.concatenate([[0], finger_timing])

    # Calculate time difference between Go Cue and start time
    #go_cue_time_diff = trial_time_milliseconds[0] - event_times[0]
    go_cue_time_diff = 0

    # Calculate adjusted event times relative to Go Cue
    adjusted_event_times = event_times + go_cue_time_diff

    # Loop through each event time and assign event labels
    for i, time_point in enumerate(adjusted_event_times):
        closest_time_index = np.argmin(np.abs(trial_time_milliseconds - time_point))
        trial_event_labels[closest_time_index] = event_labels[i]

    # Extend the event data list for the current trial
    for time, event_label in zip(trial_time_milliseconds, trial_event_labels):
        event_data.append([time, trial, event_label])

# Create a DataFrame from the event data
events_df = pd.DataFrame(event_data, columns=['Time', 'Trial Number', 'Event'])

# Save the modified data to a new CSV file called events.csv
output_csv_path = r'D:\MSc Project\Data\Old Data\MEG\s08\events.csv'
data.to_csv(output_csv_path, index=False)

# MERGE DATA

# Load the mean_power_data_python.csv file
csv_path = r'D:\MSc Project\Data\Old Data\MEG\s08\mean_power_data_python.csv'
mean_power_data = pd.read_csv(csv_path)

# Rename the columns of the mean_power_data_python.csv DataFrame
mean_power_data.columns = ['Time', 'Mean Alpha Power', 'Mean Beta Power', 'Mean Gamma Power', 'Trial Number']

# Initialize a new DataFrame for combined data
combined_data = mean_power_data.copy()

# Create an empty Events column in the combined_data DataFrame
combined_data['Events'] = ''




# Create an empty dictionary to hold event labels and times
event_times_dict = {}

# Loop through the event_times_dict
for trial_number, event_dict in event_times_dict.items():
    # Iterate through event labels and times in the current trial's dictionary
    for event_label, event_time in event_dict.items():
        # Find the index where event time is >= Time in combined_data
        index = combined_data[combined_data['Trial Number'] == trial_number][combined_data['Time'] >= event_time].index[
            0]

        # Assign the event label to the Events column at the found index in combined_data
        combined_data.at[index, 'Events'] = event_label





print("Success")



# Save the combined data to a new CSV file
combined_csv_path = r'D:\MSc Project\Data\Old Data\MEG\s08\combined_data.csv'
combined_data.to_csv(combined_csv_path, index=False)

