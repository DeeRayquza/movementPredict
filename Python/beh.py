import h5py
import numpy as np
import matplotlib.pyplot as plt

# Load the MATLAB .mat file
path = r'D:\MSc Project\Data\Old Data\Behavioural\s05\behDataMEG.mat'
mat_data = h5py.File(path)

# Function to convert HDF5 dataset to numpy array
def hdf5_to_numpy(hdf5_dataset):
    if isinstance(hdf5_dataset, h5py.Dataset):
        return np.array(hdf5_dataset)
    elif isinstance(hdf5_dataset, h5py.Group):
        return {name: hdf5_to_numpy(item) for name, item in hdf5_dataset.items()}

# Convert the entire data to Python data structures
data_dict = hdf5_to_numpy(mat_data['A'])

# X-axis
finger_timing = data_dict['timing'][:, 2]

# Marking Timestamps
for point in finger_timing:
    plt.axvline(x=point, color='r', linestyle='--')

# Y-axis
frequencies = [0, 8, 12, 30, 50]

# Plotting the graph
# plt.figure()
# plt.plot(finger_timing, frequencies, marker='o')
# plt.xlabel('Time (ms)')
# plt.ylabel('Bands (Hz)')
# plt.title('Timing of Key Presses Relative to Go Cue')
# plt.grid(True)
#
# plt.tight_layout()
# plt.show()
















'''
# PLOTTING
# Assuming you already have the data_dict with 'timing' as one of the keys
timing_data = data_dict['timing']
num_trials = len(timing_data) / 5  # Assuming there are exactly 5 datapoints per trial

# Create a dictionary to store the timing data for each trial
trials_data = {}
for i in range(num_trials):
    trial_data = timing_data[i * 5: (i + 1) * 5]
    trials_data[f'Trial_{i + 1}'] = trial_data

# Set Go cue location and plot properties
go_cue_offset = 50  # Adjust this value to control the offset
go_cue_time = np.zeros(num_trials) + go_cue_offset

# Generate trial numbers from 1 to the total number of trials
trial_numbers = np.arange(1, num_trials + 1)

# Plot the data for all trials
plt.figure(figsize=(10, 6))
for trial_num, trial_data in trials_data.items():
    # Calculate time points for each press relative to Go cue
    time_points = trial_data + go_cue_offset

    # Plot each press as a separate point
    plt.scatter(time_points, np.full_like(time_points, int(trial_num.split('_')[1])), marker='o', label=f'Trial {trial_num}')

# Plot the Go cue as a vertical line at the Go cue offset
plt.axvline(go_cue_offset, color='red', linestyle='--', label='Go Cue')

# Set plot labels and title
plt.xlabel('Time (ms)')
plt.ylabel('Trial Number')
plt.title('Timing of Key Presses Relative to Go Cue')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()


# Close the file after you are done
mat_data.close()
'''