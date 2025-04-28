import numpy as np
import scipy.io as sio
from scipy.signal import butter
import scipy.signal as signal
import matplotlib.pyplot as plt
import statsmodels.api as sm

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


def matlab_to_DL_input(mat_files, window_size, number_of_channels, sampling_freq, artifact_removal = False, filter = False):
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

                if filter:
                    # fig, axs = plt.subplots(3, 1)
                    trial = np.transpose(trial, (1, 0))
                    
                    for channel in range(number_of_channels):
                        #Plot the raw data for visualization
                        # I want to plot a single figure that contains all channels
                        time_axis = np.linspace(0, 750 / 250, 750)  # Create time axis in seconds    
                        

                        #This is the EOG data, which is the last 2 channels of the data removal of noise 
                        X_eog = mat_data["data"][0][session_num][0][0][0][trial_num: trial_num + window_size, number_of_channels:] 
                        X_eog = sm.add_constant(X_eog)
                        model_c = sm.OLS(trial[channel,:], X_eog).fit()
                        predicted_c_artifact = model_c.predict(X_eog)
                        cleaned_c = trial[channel,:] - predicted_c_artifact                 
                        #print(model_c.summary()) # Optional: Print the regression summary
                        cleaned_filtered_c = butter_bandpass_filter(cleaned_c , 8, 30, sampling_freq, 5)

                        # axs[channel].plot(time_axis, trial[channel, :], 'r') #plotting the raw data 
                        # axs[channel].plot(time_axis, cleaned_c, 'g') #plotting the raw data 
                        # axs[channel].plot(time_axis, cleaned_filtered_c, 'b') #plotting the raw data
                        
                   

                    # fig.suptitle(f" Comparison of raw and Processed Data- Class {labels[y_num]}")
                    # fig.supxlabel('Time (seconds)')
                    # fig.supylabel('Amplitude (µV)')
                    # plt.show()
                    # plt.tight_layout()
                    # plt.close()
                    trial[channel,:] = cleaned_filtered_c
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
    mat_files = ["BCI_IV_2b_mat\B01T.mat", "BCI_IV_2b_mat\B01E.mat"]
    matlab_to_DL_input(mat_files, 750, 3, sampling_freq=250, filter=True)
    # matlab_to_DL_input(mat_files, 750, 3, sampling_freq=250, filter=False)

    # mat_files = ["BCI_IV_2a_mat\A03E.mat"]
    # matlab_to_DL_input(mat_files, 500, 22)