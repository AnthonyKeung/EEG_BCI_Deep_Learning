
import numpy as np
import matplotlib.pyplot as plt 
import mne

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import  Conv2D, MaxPooling2D, Flatten, Dense, Reshape 
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split


def raw_channel_into_frame_indexes_and_labels(raw_channel, eventpos, event_dur, labels = False):

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


def gdf_to_raw_data_input(gdf_filepaths):
    # Load the GDF file
    #(750, 3 ) 750 timepoints, 3 channels
    c3_frames = np.empty((0, 750))
    c4_frames = np.empty((0, 750))
    labels = []

    for gdf_filepath in gdf_filepaths:
        raw = mne.io.read_raw_gdf(gdf_filepath, verbose=False)
        eventpos, event_dur = mne.events_from_annotations(raw)
        
        channel_index = raw.ch_names.index('EEG:C3')  # Get the index of channel 'C3'
        c3_data, _ = raw[channel_index]  # Extract the data and corresponding times
        #cz_data, _ = raw[channel_index]  # Extract the data and corresponding times
        c4_data, _ = raw[channel_index]  # Extract the data and corresponding times


        c3_frames_new, labels_new = raw_channel_into_frame_indexes_and_labels(c3_data, eventpos, event_dur, labels=True)
        c4_frames_new, _ = raw_channel_into_frame_indexes_and_labels(c4_data, eventpos, event_dur)

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

    return input_formulated_data, labels

if __name__ == '__main__':
    train_file_path = ["BCI_IV_2b/B0101T.gdf", "BCI_IV_2b/B0102T.gdf", "BCI_IV_2b/B0103T.gdf"]
    test_file_path = ["BCI_IV_2b/B0104E.gdf", "BCI_IV_2b/B0105E.gdf"]

    train_formulated_data, train_labels = gdf_to_raw_data_input(train_file_path)
    test_formulated_data, test_labels = gdf_to_raw_data_input(test_file_path)

    # Reshape the data to have a 4D shape (samples, timepoints, channels, 1)
    input_formulated_training_data = train_formulated_data[..., np.newaxis]
    input_formulated_testing_data = test_formulated_data[..., np.newaxis]

    print("Shape of input_formulated_training_data:", input_formulated_training_data.shape)  
    print("Shape of input_formulated_testing_data:", input_formulated_testing_data.shape)

    # Convert labels to categorical format
    train_labels_categorical = to_categorical(train_labels, num_classes=2)
    test_labels_categorical = to_categorical(test_labels, num_classes=2)

    # Define the CNN model
    model = Sequential([
        Conv2D(25, (11,1), activation='relu', input_shape=(750, 2, 1), strides=1, padding='valid'),
        Reshape((2, 740, 25), input_shape=(740, 2, 25)),
        Conv2D(25, (1, 2), activation='relu', input_shape=(2, 740, 25), strides=1, padding='valid'),
        MaxPooling2D((1, 3), input_shape=(1, 740,  25)),
        Reshape((246, 2, 25), input_shape=(2, 246, 25)),
        Conv2D(25, (11, 1), input_shape = (246, 2, 25), activation='relu', strides=1, padding='valid'),
        Reshape((2, 236, 25), input_shape=(236, 2, 25)), 
        MaxPooling2D((1, 3), strides=1),
        Reshape((234, 2, 25), input_shape=(2, 234, 25)), 
        Conv2D(25, (11, 1), activation='relu', strides=1, padding='valid'),
        Reshape((2 , 224, 25), input_shape=(224, 2, 25)), 
        MaxPooling2D((1, 3)),
        Reshape((74 , 2, 25), input_shape=(2, 74, 25)), 
        Conv2D(25, (11, 1), activation='relu', strides=1, padding='valid'),
        MaxPooling2D((1, 2)),
        Flatten(),
        Dense(800, activation='relu'),
        Dense(2, activation='softmax')
        ])

    # Print the model summary to see the size after each layer
    model.summary()
    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    history = model.fit(input_formulated_training_data, train_labels_categorical, epochs=10, batch_size=32, validation_data=(input_formulated_testing_data, test_labels_categorical))

    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(input_formulated_testing_data, test_labels_categorical)
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")