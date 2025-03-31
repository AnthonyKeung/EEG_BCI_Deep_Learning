
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
    input_formulated_data, labels = MI_EEG_Data.gdf_to_raw_data_input()

    input_matrix = np.zeros((6, 7, 0)) 
    single_frame = np.zeros((6, 7, window_size))
    for frame in range(len(input_formulated_data['EEG-Fz'])):

        
    
    print("Shape of input_matrix:", input_matrix.shape)

