
import numpy as np
from MI_EEG_Processor import MI_EEG_Processor

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import  Conv3D, MaxPooling3D, Flatten, Dense, Reshape 
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split


if __name__ == '__main__':
    file_paths = ["BCI_IV_2a\A01T.gdf"]
    window_size = 500
    MI_EEG_Data = MI_EEG_Processor(file_paths, window_size)
    raw_data, labels = MI_EEG_Data.gdf_to_raw_data_input()

    # # Normalise the frames to have zero mean and unit variance
    # for channel_name in raw_data.keys():
    #     raw_data[channel_name] = (raw_data[channel_name] - np.mean(raw_data[channel_name], axis=1, keepdims=True)) / np.std(raw_data[channel_name], axis=1, keepdims=True)
    

# create a 0 array of length 1,1,500
    padded_zero = np.zeros( (273, 1, window_size))
    print(padded_zero.shape)
    print(raw_data["EEG-Fz"].shape)
    input_formulated_data = np.stack((padded_zero, padded_zero, padded_zero,raw_data["EEG-Fz"], padded_zero, padded_zero, padded_zero), axis=-1)