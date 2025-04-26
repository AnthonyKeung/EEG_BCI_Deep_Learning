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

        # This is to check if number of LH and RH labels are similar 
        LH_total_count = 0
        RH_total_count = 0

        events_extracted_full = []

        
        for gdf_filepath in self.gdf_filepaths:
            raw = mne.io.read_raw_gdf(gdf_filepath, verbose=False)
            eventpos, event_dur = mne.events_from_annotations(raw)


            # Get the labels for a single filepath it's an extra loop but only once 
            channel_data, _ = raw[raw.ch_names[0]]
            _, labels_new, LH_labels, RH_labels, event_pos_extracted = self.__raw_channel_into_frame_indexes_and_labels(channel_data, eventpos, event_dur)

            labels.extend(labels_new)
            events_extracted_full.extend(event_pos_extracted)
            LH_total_count += LH_labels
            RH_total_count += RH_labels
            # print(raw.ch_names)

            for channel_name in raw.ch_names:
                channel_index = raw.ch_names.index(channel_name)  
                channel_data, _ = raw[channel_index]  
                frames_new, _, _, _, _= self.__raw_channel_into_frame_indexes_and_labels(channel_data, eventpos, event_dur)
                proccessed_data[channel_name]= np.vstack((proccessed_data[channel_name], frames_new))
                
                


        print("label length is ", len(labels))
        #This is to check if the labels are correct
        # Get 10 random indices from the labels list for checking
        random_indices = np.random.randint(0, len(labels), size=10).tolist()

        for i in random_indices:
            print("Random labels from extracted data is ", labels[i])
            print("Random labels from raw data is ", events_extracted_full[i])

        print("LH labels count is ", LH_total_count)
        print("RH labels count is ", RH_total_count)
        print("Shape of input_formulated_data:", proccessed_data[raw.ch_names[0]].shape)

        return proccessed_data, labels


    def __raw_channel_into_frame_indexes_and_labels(self, raw_channel, eventpos, event_dur):
        # print(eventpos[0:30])

        start_trial_number = event_dur["768"] # 768 is the start of the trial
        Rejected_trial_number = event_dur["1023"] # 1023 is the rejected trial

        # Find the index of the first trial start event
        start_trial_index = np.argmax(eventpos[:, 2] == start_trial_number)

        # Iterate through the event positions
        channel_frames = np.empty((0, self.window_size))
        channel_labels = []
        current_trial = start_trial_index

        LH_labels = 0
        RH_labels = 0
        foot_labels = 0
        tongue_labels = 0

        # Check if the lables are labelled correctly
        event_pos_extracted = []

        while (current_trial < len(eventpos[:,0])):
            # Check for trial start 
            if eventpos[current_trial , 2] == start_trial_number:

                # Check if the next trial is an ivalid trial
                if eventpos[current_trial+1, 2] == Rejected_trial_number:
                    current_trial += 3
                    
                else:
                    trial_start_index = eventpos[current_trial , 0]
                    trial_end_index = eventpos[current_trial+1, 0] if current_trial+1 < len(eventpos) else len(raw_channel)

                    # print("Trial start index is ", trial_start_index)
                    # print("Trial end index is ", trial_end_index)

                    # Extract the trial data
                    trial_data = raw_channel[0, trial_start_index:trial_end_index]
                    channel_frames = np.vstack((channel_frames, trial_data))

                    channel_label = int(list(event_dur.keys())[list(event_dur.values()).index(eventpos[current_trial+1, 2])])
                    channel_labels.append(channel_label)

                    # Check to see if the actual extracted data is correct
                    pseudo_event_pos = eventpos[current_trial+1, :].copy()
                    pseudo_event_pos[2] = channel_label
                    event_pos_extracted.append(pseudo_event_pos)
                    

                    if channel_label  == 769:
                        LH_labels += 1
                    elif channel_label  == 770:
                        RH_labels += 1
                    elif channel_label  == 771:
                        foot_labels += 1
                    elif channel_label  == 772:
                        tongue_labels += 1
                    else:
                        raise ValueError("Unknown label encountered.")
                    
                    current_trial += 2
                    
            else:
                current_trial += 1

        if len(channel_frames) != len(channel_labels):
            raise ValueError("Mismatch between the number of channel frames and channel labels.")
        
        return channel_frames,channel_labels, LH_labels, RH_labels, event_pos_extracted
    
if __name__ == "__main__":
    gdf_filepaths = ["BCI_IV_2a\A01T.gdf"]
    window_size = 500  
    processor = MI_EEG_Processor(gdf_filepaths, window_size)
    processed_data, labels = processor.gdf_to_raw_data_input()
    
    
