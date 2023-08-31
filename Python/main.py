# Importing the Libraries
import h5py
import numpy as np
import mne
import scipy.io
import matplotlib.pyplot as plt
from scipy.signal import spectrogram

# Load .mat MATLAB data file
file_path = r'D:\MSc Project\Data\Old Data\MEG\s05\BfcPRdmg4670_blk02_ICA_corrected.mat'
mat_data = scipy.io.loadmat(file_path, squeeze_me=True, struct_as_record=False)


# Convert mat_struct object to nested Python dictionary
def convert_to_dict(obj):
    if isinstance(obj, scipy.io.matlab.mat_struct):
        # Convert MATLAB struct to nested dictionary
        data = {}
        for field_name in obj._fieldnames:
            field_value = getattr(obj, field_name)
            data[field_name] = convert_to_dict(field_value)
        return data
    elif isinstance(obj, dict):
        # Handle the case where the object is already a dictionary
        data = {}
        for key, value in obj.items():
            data[key] = convert_to_dict(value)
        return data
    else:
        # For non-dictionary and non-struct objects, return the value as-is
        return obj

nested_dict = convert_to_dict(mat_data)

# View the nested dictionary as a tree
def print_dict_as_tree(d, indent=0):
    for key, value in d.items():
        if isinstance(value, dict):
            print('  ' * indent + f'{key}:')
            print_dict_as_tree(value, indent + 1)
        else:
            print('  ' * indent + f'{key}: {value}')

#print_dict_as_tree(nested_dict)
#print(nested_dict['D'])

# Print all keys in the nested dictionary
def print_all_keys(d, prefix=''):
    for key, value in d.items():
        if isinstance(value, dict):
            print_all_keys(value, prefix + key + '.')
        else:
            print(prefix + key)

# View all keys in the nested dictionary
#print_all_keys(nested_dict)

#print("KEYS FINISHED")

# Plotting Time Frequency Representation

# Accessing MEG data from the nested dictionary
meg_data = nested_dict['D']['data']

# Accessing sampling frequency from the nested dictionary
sampling_frequency = nested_dict['D']['Fsample']

# Time information (replace with the actual time array)
time_axis = np.linspace(0, meg_data.shape[1] / sampling_frequency, meg_data.shape[1])

# Frequency parameters (replace with the desired frequency bins)
frequency_bins = np.logspace(1, 3, 100)  # Example: logarithmically spaced bins from 10 Hz to 1000 Hz

# Time-Frequency analysis using STFT
frequencies, times, tfr = spectrogram(
    meg_data, fs=sampling_frequency, nperseg=256, noverlap=128, nfft=512
)

# Plot the TFR
plt.figure(figsize=(10, 6))
plt.pcolormesh(times, frequencies, np.log10(tfr), shading='auto')
plt.colorbar(label='Log Power Spectral Density (dB)')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.title('Time-Frequency Representation (TFR)')
plt.show()

