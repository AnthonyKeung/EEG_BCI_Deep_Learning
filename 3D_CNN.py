
import numpy as np
from MI_EEG_Processor import MI_EEG_Processor

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import  Conv3D, MaxPooling3D, Flatten, Dense, Reshape 
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split


if __name__ == '__main__':
    file_paths = ["BCI_IV_2b/B0101T.gdf", "BCI_IV_2b/B0102T.gdf", "BCI_IV_2b/B0103T.gdf"]
    window_size = 750
    MI_EEG_Data = MI_EEG_Processor(file_paths, window_size)
    input_formulated_data, labels = MI_EEG_Data.gdf_to_raw_data_input()