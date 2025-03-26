# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import numpy as np
import matplotlib.pyplot as plt

import mne

# Path to the .gdf file
file_path = "BCI_IV_2b/B0104E.gdf"

# Load the GDF file
raw = mne.io.read_raw_gdf(file_path, verbose=True )

# print("Information",raw.info)
# print("Channel Names:",raw.ch_names)
# raw.plot()
# plt.show()

# ##Input formula for 3D CNN:
# # Extract dat for channel 'C3'
# channel_index = raw.ch_names.index('EEG:C3')  # Get the index of channel 'C3'
# c3_data, times = raw[channel_index]  # Extract the data and corresponding times
# # print("C3 Data Shape:", c3_data.shape) ##C3 Data Shape: (1, 604803)
# # print("Type of C3 Data:", type(c3_data)) ##tis a numpy array
# # print("C3 Data:", c3_data) ## [[ 4.15045396e-06  4.44647898e-06  2.20035096e-06 ...  2.11795224e-06, 1.40993362e-06 -6.71396963e-07]]

# channel_index = raw.ch_names.index('EEG:Cz')  # Get the index of channel 'C3'
# cz_data, _ = raw[channel_index]  # Extract the data and corresponding times

# channel_index = raw.ch_names.index('EEG:C4')  # Get the index of channel 'C3'
# cz_data, _ = raw[channel_index]  # Extract the data and corresponding times

# # Create a 3D numpy array with the desired structure
# # Initialize a 3D array of zeros with shape (3, 3, timepoints)
# timepoints = c3_data.shape[1]  # Get the number of timepoints
# kernel_3d = np.zeros((3, 3, timepoints))

# # Fill the center row with C3, Cz, and C4 data
# kernel_3d[1, 0, :] = c3_data  # C3 data #means second , first column , all timepoints
# kernel_3d[1, 1, :] = cz_data  # Cz data
# kernel_3d[1, 2, :] = cz_data  # C4 data

# print("3D Kernel Shape:", kernel_3d.shape)
# print("3D Kernel:", kernel_3d[:,:,0])

#########CNN Labeling##
# #Figure out how to attach labels to the data 
eventpos, event_dur = mne.events_from_annotations(raw, verbose=True)
#[np.str_('1023'), np.str_('1077'), np.str_('1078'), np.str_('1079'), np.str_('1081'), np.str_('276'), np.str_('277'), np.str_('32766'), np.str_('768'), np.str_('769'), np.str_('770')]
#[1,2,3,4,5,6,7,8,9,10,11]

print("Event durations:", event_dur["768"])
print("type of event_dur:", type(event_dur))
print("size of eventpos", eventpos.shape)
print("Bottom 60 results of eventpos:")
print(eventpos[-30:])

# print(type(eventpos))
# first_instance_index = np.argmax(eventpos[:, 2] == 9)
# print("First instance of 9 in the 3rd column is at index:", first_instance_index)
# # I know 

# #####