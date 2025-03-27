import numpy as np
import matplotlib.pyplot as plt 
import mne

class MI_EEG_Processor:
    """
    This class is responsible for processing the BCI competition data into frames and labels.
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

        #first create an empty dictionary with the key being the channel name and the value being the frames
        raw = mne.io.read_raw_gdf(self.gdf_filepaths[0], verbose=False)
        proccessed_data = {channel_name: np.empty((0, 750)) for channel_name in raw.ch_names}
        labels = []
        
        for gdf_filepath in self.gdf_filepaths:
            raw = mne.io.read_raw_gdf(gdf_filepath, verbose=False)
            eventpos, event_dur = mne.events_from_annotations(raw)
            do_I_need_label = True
            print("Channel_names" ,raw.ch_names)

            for channel_name in raw.ch_names:
                channel_index = raw.ch_names.index(channel_name)  # Get the index of channel 'C3'
                channel_data, _ = raw[channel_index]  # Extract the data and corresponding times

                frames_new, labels_new = self.__raw_channel_into_frame_indexes_and_labels(channel_data, eventpos, event_dur, labels=do_I_need_label)

                proccessed_data[channel_name]= np.vstack((proccessed_data[channel_name], frames_new))
                labels.extend(labels_new)


                # Normalize the frames to have zero mean and unit variance
                proccessed_data[channel_name] = (proccessed_data[channel_name] - np.mean(proccessed_data[channel_name], axis=1, keepdims=True)) / np.std(proccessed_data[channel_name], axis=1, keepdims=True)
                do_I_need_label = False
        print("label length is ", len(labels))
        print("Shape of input_formulated_data:", proccessed_data['EEG:Cz'].shape)
        return proccessed_data, labels
            #     # Ensure all channels have the same length
            #     if not (len(c3_frames) == len(c4_frames)):
            #         raise ValueError("Channel data lengths are not equal. Ensure all channels have the same length.")
                
            # # Stack the channels into a 2D array (timepoints, channels)
            # input_formulated_data = np.stack((c3_frames, c4_frames), axis=-1)

            # # Reshape the data to have a 4D shape (samples, timepoints, channels, 1)
            # input_formulated_data = input_formulated_data[..., np.newaxis]
            # print("Shape of input_formulated_data:", input_formulated_data.shape)   

            # return input_formulated_data, labels