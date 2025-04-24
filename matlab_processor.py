import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from mne.decoding import CSP
import scipy.io as sio

def matlab_to_DL_input(mat_files):
    """
    Convert MATLAB data to a format suitable for deep learning.
    
    Parameters:
    mat_file (list): List of paths to MATLAB files.
    
    Returns:
    np.ndarray: Formatted data ready for deep learning.
    """
    x = []
    y = []
    

    for mat_file in mat_files:
        mat_data = sio.loadmat(mat_file)
        number_of_sessions = mat_data["data"].shape[1]
        print("Number of sessions:", number_of_sessions)
        print("Keys in the .mat file:", mat_data.keys())

        # get data in      X : array, shape (n_epochs, n_channels, n_times) format 
        for session_num in range(number_of_sessions):
            labels = mat_data["data"][0][session_num][0][0][2].flatten()
            print(labels)
            y.extend(labels)
            for trial_num in mat_data["data"][0][session_num][0][0][1]:
                trial_num  = int(trial_num[0])
                trial  = mat_data["data"][0][session_num][0][0][0][trial_num-1: trial_num + 749, 0:3]
                x.append(trial)

        

        print("Shape of x:", np.shape(x))
        print("Shape of y:", len(y))

        x_reshaped = np.transpose(x, (0, 2, 1))
        
        print("Shape of x_reshaped:", np.shape(x_reshaped))


          
if __name__ == "__main__":
    mat_files = ["BCI_IV_2b_mat\B01T.mat"]
    matlab_to_DL_input(mat_files)