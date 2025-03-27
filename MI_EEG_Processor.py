import numpy as np
import matplotlib.pyplot as plt 
import mne

class MI_EEG_Processor:
    """
    This class is responsible for processing the EEG data from the MI EEG dataset into frames and labels.
    """
    def __init__(self, gdf_filepaths):
        self.gdf_filepaths = gdf_filepaths
        pass

    def __raw_channel_into_frame_indexes_and_labels(self, raw_channel, eventpos, event_dur, labels=False):

        start_trial_number = event_dur["768"] # 768 is the start of the trial
        left_hand_trial_number = event_dur["769"] # 769 is the left hand trial
        right_hand_trial_number = event_dur["770"] # 770 is the right hand trial
        Rejected_trial_number = event_dur["1023"] # 1023 is the rejected trial

        # Find the index of the first trial start event
        start_trial_index = np.argmax(eventpos[:, 2] == start_trial_number)

        # Iterate through the event positions
        channel_frames = np.empty((0, 750))
        channel_labels = []
        current_trial = start_trial_index

        while (current_trial < len(eventpos[:,0])):
            # Extract the start and end indices of the trial
            # print("i:",current_trial)
            # print("Trial Start:", eventpos[current_trial , 0])
            # print("Trial End Index:", eventpos[current_trial+1, 0])
            # print("Window size is ", eventpos[current_trial+1, 0] - eventpos[current_trial , 0])

            # Check if the current trial is a valid trial 
            if eventpos[current_trial , 2] == start_trial_number:

                # Check if the next trial is an ivalid trial
                if eventpos[current_trial+1, 2] == Rejected_trial_number:
                    current_trial += 3
                    
                else:
                    trial_start_index = eventpos[current_trial , 0]
                    trial_end_index = eventpos[current_trial+1, 0] if current_trial+1 < len(eventpos) else len(raw_channel)

                    # Extract the trial data
                    trial_data = raw_channel[0, trial_start_index:trial_end_index]
                    channel_frames = np.vstack((channel_frames, trial_data))

                    # Extract the trial label
                    if labels:
                        if eventpos[current_trial+1, 2] == left_hand_trial_number:
                            channel_labels.append(0) # Left hand == 0
                        elif eventpos[current_trial+1, 2] == right_hand_trial_number:
                            channel_labels.append(1) # Right hand == 1
                        else:
                            raise ValueError("Invalid event Type. Expected 10 or 11.")
                    current_trial += 2
            else:
                current_trial += 1


        if labels:
            if len(channel_frames) != len(channel_labels):
                raise ValueError("Mismatch between the number of channel frames and channel labels.")
        
        return channel_frames,channel_labels
    
    def gdf_to_raw_data_input(self):
        # Load the GDF file
        #(750, 3 ) 750 timepoints, 3 channels
        c3_frames = np.empty((0, 750))
        c4_frames = np.empty((0, 750))
        labels = []

        for gdf_filepath in self.gdf_filepaths:
            raw = mne.io.read_raw_gdf(gdf_filepath, verbose=False)
            eventpos, event_dur = mne.events_from_annotations(raw)
            
            channel_index = raw.ch_names.index('EEG:C3')  # Get the index of channel 'C3'
            c3_data, _ = raw[channel_index]  # Extract the data and corresponding times
            #cz_data, _ = raw[channel_index]  # Extract the data and corresponding times
            c4_data, _ = raw[channel_index]  # Extract the data and corresponding times


            c3_frames_new, labels_new = self.__raw_channel_into_frame_indexes_and_labels(c3_data, eventpos, event_dur, labels=True)
            c4_frames_new, _ = self.__raw_channel_into_frame_indexes_and_labels(c4_data, eventpos, event_dur)

            c3_frames = np.vstack((c3_frames, c3_frames_new))
            c4_frames = np.vstack((c4_frames, c4_frames_new))
            print("Number of frames ",len(labels_new))
            labels.extend(labels_new)


            # Normalize the frames to have zero mean and unit variance
            c3_frames = (c3_frames - np.mean(c3_frames, axis=1, keepdims=True)) / np.std(c3_frames, axis=1, keepdims=True)
            #cz_frames = (cz_frames - np.mean(cz_frames, axis=1, keepdims=True)) / np.std(cz_frames, axis=1, keepdims=True)
            c4_frames = (c4_frames - np.mean(c4_frames, axis=1, keepdims=True)) / np.std(c4_frames, axis=1, keepdims=True)

            # Ensure all channels have the same length
            if not (len(c3_frames) == len(c4_frames)):
                raise ValueError("Channel data lengths are not equal. Ensure all channels have the same length.")
            
        # Stack the channels into a 2D array (timepoints, channels)
        input_formulated_data = np.stack((c3_frames, c4_frames), axis=-1)

        # Reshape the data to have a 4D shape (samples, timepoints, channels, 1)
        input_formulated_data = input_formulated_data[..., np.newaxis]
        print("Shape of input_formulated_data:", input_formulated_data.shape)   

        return input_formulated_data, labels