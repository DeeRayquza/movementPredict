import mne
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

path = 'D:\MSc Project\Data\Old Data\MEG\s05\BfcPRdmg4670_blk02_ICA_corrected.mat'

# Load the .mat file using scipy.io.loadmat
mat_data = sio.loadmat(path)

# Extract the raw data array ('D') from the loaded .mat file
data = mat_data['D']['data'][0, 0]

# Transpose the data array to have shape (channels x time points)
data = np.transpose(data)

# Get the channel names from the loaded .mat file and flatten the array
channel_names = mat_data['D']['channels'][0, 0]['label'].squeeze()
channel_names = [name[0] for name in channel_names]

print("Channel Names:", channel_names)

# Map 'MEGGRAD' to 'grad' as the channel type
channel_type = 'grad'

# Get the sensor positions from the loaded .mat file and convert to 1D arrays using flatten()
x_pos = mat_data['D']['channels'][0, 0]['X_plot2D'].flatten()
y_pos = mat_data['D']['channels'][0, 0]['Y_plot2D'].flatten()

# Filter out empty arrays and keep only valid sensor positions
valid_idx = np.where([pos.size != 0 for pos in x_pos])[0]
x_pos = np.array([x_pos[i][0] for i in valid_idx])
y_pos = np.array([y_pos[i][0] for i in valid_idx])

# Print the shapes of x_pos and y_pos after filtering
print("Shape of x_pos after filtering:", x_pos.shape)
print("Shape of y_pos after filtering:", y_pos.shape)

# Create a scatter plot of the sensor positions
plt.figure()
plt.scatter(x_pos, y_pos, s=30, color='blue')
for name, x, y in zip(channel_names, x_pos, y_pos):
    plt.text(x, y, name, fontsize=8, ha='center', va='center')
plt.xlabel('X position (mm)')
plt.ylabel('Y position (mm)')
plt.title('Sensor Positions')

# # Create an info object with 'grad' channel types and sensor positions
# ch_types = ['grad'] * len(channel_names)
# info = mne.create_info(ch_names=channel_names, sfreq=1000, ch_types=ch_types)
#
# # Create a layout object with the sensor positions
# layout = mne.channels.make_grid_layout(info)
#
# # Create a topographic view of the sensor positions (using zeros for data as an example)
# plt.figure()
# mne.viz.plot_topomap(data=np.zeros(len(channel_names)), pos=layout.pos, names=channel_names, sphere=0.1, contours=0, outlines='head')
plt.show()









# # Create a scatter plot of the sensor positions using plotly
# fig = go.Figure()
# fig.add_trace(go.Scatter(x=x_pos, y=y_pos, mode='markers', marker=dict(size=8), text=channel_names, hoverinfo='text'))
# fig.update_layout(title='Sensor Positions', xaxis_title='X position (mm)', yaxis_title='Y position (mm)')
#
# fig.show()






# # Create an MNE-Python 'info' dictionary with the required information
# info = mne.create_info(ch_names=channel_names, sfreq=1000, ch_types='grad')
#
# # Create an MNE-Python 'Raw' object using the data and info
# raw = mne.io.RawArray(data, info)
#
# # Extract information from the loaded data and store it in the 'meg' dictionary
# meg = {
#     'data': raw.get_data(),         # Raw data (numpy array: channels x time points)
#     'info': raw.info,               # Information about channels and sampling rate
#     'times': raw.times,             # Time points of the data
#     'channel_names': raw.ch_names,  # Channel names
# }
#
# # You can now access the information using the 'meg' dictionary
# print(meg['info'])
# print(meg['times'])
# print(meg['channel_names'])
# print(meg['data'].shape)
