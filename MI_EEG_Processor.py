import numpy as np
import matplotlib.pyplot as plt 
import mne

class MI_EEG_Processor:
    """
    This class is responsible for processing the BCI competition data into frames and labels.
    """
    def __init__(self, gdf_filepaths, window_size):
        self.gdf_filepaths = gdf_filepaths
        self.window_size = window_size
        pass

    def gdf_to_raw_data_input(self):

        #first create an empty dictionary with the key being the channel name and the value being the frames
        raw = mne.io.read_raw_gdf(self.gdf_filepaths[0], verbose=False)
        proccessed_data = {channel_name: np.empty((0, self.window_size)) for channel_name in raw.ch_names}
        labels = []
        
        for gdf_filepath in self.gdf_filepaths:
            raw = mne.io.read_raw_gdf(gdf_filepath, verbose=False)
            eventpos, event_dur = mne.events_from_annotations(raw)
            print("Channel_names" ,raw.ch_names)

            # Get the labels for a single filepath it's an extra loop but only once 
            channel_data, _ = raw[raw.ch_names[0]] ###this may cause issues if the channel name does not exist
            _, labels_new = self.__raw_channel_into_frame_indexes_and_labels(channel_data, eventpos, event_dur)
            labels.extend(labels_new)

            for channel_name in raw.ch_names:
                channel_index = raw.ch_names.index(channel_name)  
                channel_data, _ = raw[channel_index]  
                frames_new, _ = self.__raw_channel_into_frame_indexes_and_labels(channel_data, eventpos, event_dur)
                proccessed_data[channel_name]= np.vstack((proccessed_data[channel_name], frames_new))
                
                

        print("label length is ", len(labels))
        print("Shape of input_formulated_data:", proccessed_data[raw.ch_names[0]].shape)

        return proccessed_data, labels


    def __raw_channel_into_frame_indexes_and_labels(self, raw_channel, eventpos, event_dur):

        start_trial_number = event_dur["768"] # 768 is the start of the trial
        Rejected_trial_number = event_dur["1023"] # 1023 is the rejected trial

        # Find the index of the first trial start event
        start_trial_index = np.argmax(eventpos[:, 2] == start_trial_number)

        # Iterate through the event positions
        channel_frames = np.empty((0, self.window_size))
        channel_labels = []
        current_trial = start_trial_index

        while (current_trial < len(eventpos[:,0])):
            # Check for trial start 
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
                    channel_labels.append(eventpos[current_trial+1, 2])
                    current_trial += 2
                    
            else:
                current_trial += 1

            if len(channel_frames) != len(channel_labels):
                raise ValueError("Mismatch between the number of channel frames and channel labels.")
        
        return channel_frames,channel_labels
    
    
