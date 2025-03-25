
import numpy as np
import matplotlib.pyplot as plt 
import mne

def raw_channel_into_frames(raw_channel, eventpos):
    #[[     0      0      8]
    #  [   499      0      6]
    #  [ 16749      0      7]
    #  [ 32999      0      5]
    #  [ 37999      0      4]
    #  [ 42999      0      3]
    #  [ 47999      0      2]
    #  [ 52251      0      8]
    #  [ 55139      0      9]
    # Find the first instance of 9 in the 3rd column
    start_trial_index = np.argmax(eventpos[:, 2] == 9)

    # Iterate through the event positions
    channel_frames = np.empty((0, 750))
    for i in range(len(eventpos[start_trial_index,:])):
        # Extract the start and end indices of the trial
        i = i*2 + start_trial_index
        print("i:",i)
        trial_start_index = eventpos[i , 0]
        print("Trial Start:", trial_start_index)
        trial_end_index = eventpos[i+1, 0] if i+1 < len(eventpos) else len(raw_channel)
        print("Trial End Index:", trial_end_index) 
        print("Window size is ", trial_end_index - trial_start_index)

        # Extract the trial data
        trial_data = raw_channel[0, trial_start_index:trial_end_index]

        print("Trial Data Shape:", trial_data.shape)
        # Append the trial data to the 2D array
        channel_frames = np.vstack((channel_frames, trial_data))
    
    print("Channel Size",channel_frames.shape)
    return channel_frames



def gdf_to_raw_data_input(gdf_filepath):
    # Load the GDF file
    #(750, 3 ) 750 timepoints, 3 channels
    raw = mne.io.read_raw_gdf(gdf_filepath, verbose=False)
    eventpos, event_dur = mne.events_from_annotations(raw)
    
    channel_index = raw.ch_names.index('EEG:C3')  # Get the index of channel 'C3'
    c3_data, _ = raw[channel_index]  # Extract the data and corresponding times
    # print("C3 Data Shape:", c3_data.shape) ##C3 Data Shape: (1, 604803)
    #print("Type of C3 Data:", type(c3_data)) ##tis a numpy array
    # print("C3 Data:", c3_data) ## [[ 4.15045396e-06  4.44647898e-06  2.20035096e-06 ...  2.11795224e-06, 1.40993362e-06 -6.71396963e-07]]

    channel_index = raw.ch_names.index('EEG:Cz')  # Get the index of channel 'C3'
    cz_data, _ = raw[channel_index]  # Extract the data and corresponding times

    channel_index = raw.ch_names.index('EEG:C4')  # Get the index of channel 'C3'
    c4_data, _ = raw[channel_index]  # Extract the data and corresponding times

    raw_channel_into_frames(c3_data, eventpos)

    # # Ensure all channels have the same length
    # if not (len(c3_data[0]) == len(cz_data[0]) == len(c4_data[0])):
    #     raise ValueError("Channel data lengths are not equal. Ensure all channels have the same length.")
    
    # # Stack the channels into a 2D array (timepoints, channels)
    # data = np.stack([c3_data[0], cz_data[0], c4_data[0]], axis=1)  # Shape: (timepoints, 3)
    # print(type(data))
    # print(data.shape)
    # print(data)
    
    # return data

if __name__ == '__main__':
    file_path = "BCI_IV_2b/B0101T.gdf"
    raw_data = gdf_to_raw_data_input(file_path)

    # [ 47999      0      2]
    # [ 52251      0      8]
    # [ 55139      0      9]
    # [ 55889      0     10]
    # [ 57499      0      9]
    # [ 58249      0     11]
    # [ 62067      0      9]
    # [ 62817      0     10]
    # [ 64104      0      9]
    # [ 64854      0     11]
    # [ 72981      0      9]
    # [ 73731      0     11]
    # [ 75188      0      9]
    # [ 75938      0     11]
    # [ 77589      0      9]