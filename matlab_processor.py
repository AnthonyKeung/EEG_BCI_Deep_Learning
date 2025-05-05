import numpy as np
import scipy.io as sio
from scipy.signal import butter
import scipy.signal as signal
import matplotlib.pyplot as plt
import statsmodels.api as sm
import os

from scipy.ndimage import shift, zoom
import random

#matlab preprocesser 
#Some filtering to remove noise from the data
def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='bandpass')
    return b, a

def shifting(data, shift_range=(-5, 5)):
    """Shifts the data along the time axis."""
    shift_val = np.random.randint(shift_range[0], shift_range[1])
    return shift(data, shift=(0, shift_val), mode='nearest') # Assuming last dimension is time


def matlab_to_DL_input(mat_files, window_size, number_of_channels, sampling_freq, filter = False, normalise = True):
    """
    Convert MATLAB data to a format suitable for deep learning.
    
    Parameters:
    mat_file (list): List of paths to MATLAB files.
    
    Returns:
    np.ndarray: Formatted data ready for deep learning  (n_trials, n_channels, n_timepoints)
    """
    x = []
    y = []
    
    for mat_file in mat_files:
        mat_data = sio.loadmat(mat_file)
        number_of_sessions = mat_data["data"].shape[1]
        print("Number of sessions:", number_of_sessions)

        # get data in      X : array, shape (n_epochs, n_channels, n_times) format 
        for session_num in range(number_of_sessions):
            labels = mat_data["data"][0][session_num][0][0][2].flatten()
            y.extend(labels)
            for y_num, trial_num in enumerate(mat_data["data"][0][session_num][0][0][1]):
                trial_num  = int(trial_num[0])
                trial  = mat_data["data"][0][session_num][0][0][0][trial_num: trial_num + window_size, 0:number_of_channels] 
                if normalise == True: 
                    trial = (trial - np.mean(trial, axis=(0, 1), keepdims=True)) / np.std(trial, axis=(0, 1), keepdims=True)  #(n_timepoints, n_channels )
                if filter:
                    trial = np.transpose(trial, (1, 0)) # from (n_times, n_channels) to (n_channels, n_times)

                    #This is the EOG data, which is the last 2 channels of the data removal of noise 
                    X_eog = mat_data["data"][0][session_num][0][0][0][trial_num: trial_num + window_size, number_of_channels:] 
                    X_eog = sm.add_constant(X_eog)
     
                    for channel in range(number_of_channels):
                        model_c = sm.OLS(trial[channel,:], X_eog).fit()
                        predicted_c_artifact = model_c.predict(X_eog)  
                        cleaned_c = trial[channel,:] - predicted_c_artifact                 
                        #print(model_c.summary()) # Optional: Print the regression summary
                        cleaned_filtered_c = butter_bandpass_filter(cleaned_c , 8, 30, sampling_freq, 5)
                        trial[channel,:] = cleaned_filtered_c
 
                        # axs[channel].plot(time_axis, trial[channel, :], 'r') #plotting the raw data 
                        # axs[channel].plot(time_axis, cleaned_c, 'g') #plotting the raw data 
                        # axs[channel].plot(time_axis, cleaned_filtered_c, 'b') #plotting the raw data
                        
                   

                    # fig.suptitle(f" Comparison of raw and Processed Data- Class {labels[y_num]}")
                    # fig.supxlabel('Time (seconds)')
                    # fig.supylabel('Amplitude (µV)')
                    # plt.show()
                    # plt.tight_layout()
                    # plt.close()
                    
                    x.append(trial)
                else:
                    x.append(trial)

        
    x = np.array(x)
    print("Shape of x:", np.shape(x))
    print(type(x))
    print("Shape of y:", len(y))

    if not filter:
        x = np.transpose(x, (0, 2, 1))

    return x, y


          
if __name__ == "__main__":
    # save_path = "C:\Bath\Year 5\FYP\LightConvNet-main\dataset/bci_iv_2b/raw"
    # for i in range (1,10):
    #     mat_files = [ f"BCI_IV_2b_mat\B0{i}E.mat"]
    #     X, Y = matlab_to_DL_input(mat_files, 750, 3, sampling_freq=250)
    #     X = X[:, :, :-1]
    #     print("size of X ", X.shape)
    #     np.save(os.path.join(save_path, f'B0{i}E_label.npy'), Y)
    #     np.save(os.path.join(save_path, f'B0{i}E_data.npy'), X)



    
    # matlab_to_DL_input(mat_files, 750, 3, sampling_freq=250, filter=False)

    mat_files = ["BCI_IV_2a_mat\A03E.mat"]
    matlab_to_DL_input(mat_files, 500, 22, sampling_freq=250, filter=True)