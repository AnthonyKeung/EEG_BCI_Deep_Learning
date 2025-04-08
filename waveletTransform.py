import numpy as np
import matplotlib.pyplot as plt
from MI_EEG_Processor import MI_EEG_Processor
import pywt
from scipy.signal import butter, lfilter
import scipy.signal as signal

#load the data 
file_paths = ["BCI_IV_2b/B0201T.gdf", "BCI_IV_2b/B0202T.gdf", "BCI_IV_2b/B0203T.gdf"]
window_size = 750
num_of_classes = 2


MI_EEG_Data = MI_EEG_Processor(file_paths, window_size)
processed_dictionary_data, labels = MI_EEG_Data.gdf_to_raw_data_input()

#Plot the amplittude of the data against time 
sampling_rate = 250 #hz
t = np.linspace(0, len(processed_dictionary_data["EEG:C3"][0])/sampling_rate, len(processed_dictionary_data["EEG:C3"][0]))
C3 = processed_dictionary_data["EEG:C3"][0]
C4 = processed_dictionary_data["EEG:C4"][0]
plt.plot(t, C3, 'r', label='C3')
plt.plot(t, C4, 'b', label='C4')
plt.xlabel('time /s')
plt.ylabel('amplitude')
plt.legend(loc='upper right',fontsize='large',frameon=True,edgecolor='blue') 
plt.show()

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


'''
Bandpass filter to all the frames in C3 and C4 channels
'''

filtered_C3_data = []
filtered_C4_data = []
for frame in processed_dictionary_data["EEG:C3"]:
    filtered_C3_data.append(butter_bandpass_filter(frame, 8, 30, sampling_rate, 5))
for frame in processed_dictionary_data["EEG:C4"]:
    filtered_C4_data.append(butter_bandpass_filter(frame, 8, 30, sampling_rate, 5))

print("Shape of filtered_C3_data:", np.array(filtered_C3_data).shape)
print("Shape of filtered_C4_data:", np.array(filtered_C4_data).shape)


'''
EEG FFT compare between filter and filter before
'''
random_index = np.random.randint(0, len(processed_dictionary_data["EEG:C3"]), size=1)[0] # Random index for C3 channel
# FFT transfrom
dataFtt = np.fft.fft(processed_dictionary_data["EEG:C3"][random_index])  # C3 channel
Freq = np.linspace(1/len(processed_dictionary_data["EEG:C3"][random_index]),sampling_rate,len(processed_dictionary_data["EEG:C3"][random_index])) 

# plot FFT
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
ax1.plot(Freq, np.abs(dataFtt), color='blue')
ax1.set_title(f'C3 Frame {random_index} Channel Filter Before')
plt.xlabel('Freq (in Hz)')
plt.ylabel('Amplitude')

#after filtering 
dataFtt = np.fft.fft(filtered_C3_data[random_index])  # C3 channel
ax2.plot(Freq, np.abs(dataFtt), color='green')
ax2.set_title(f'C3 Channel Frame {random_index} Filter After')
plt.show()

##########Wavelet transform##############

t = np.arange(0, 3.0, 1.0/sampling_rate)
wavename = 'morl'   
totalscal = 64    # scale 
fc = pywt.central_frequency(wavename) #  central frequency
cparam = 2 * fc * totalscal
scales = cparam/np.arange(1,totalscal+1)

# C3 channel
[cwtmatr, frequencies] = pywt.cwt(filtered_C3_data[random_index], scales, wavename, 1.0/sampling_rate) # continuous wavelet transform
fig = plt.figure(1)
plt.title(f'C3 Channel Frame {random_index}')
plt.contourf(t, frequencies, abs(cwtmatr))
plt.ylabel(u"freq(Hz)")
plt.xlabel(u"time(s)")
plt.colorbar()
fig = plt.figure(2)

# C4 channel
[cwtmatr, frequencies] = pywt.cwt(filtered_C4_data[random_index],scales,wavename,1.0/sampling_rate) #
plt.title(f'C4 Channel Frame {random_index}')
plt.contourf(t, frequencies, abs(cwtmatr))
plt.ylabel(u"freq(Hz)")
plt.xlabel(u"time(s)")
plt.colorbar()
plt.show()

# ####Actual Transformation######
# totalscal = 64    # scale 
# fc = pywt.central_frequency(wavename) #  central frequency
# cparam = 2 * fc * totalscal
# scales = cparam/np.arange(1,totalscal+1)
# t = np.arange(0, 3, 1.0/sampling_rate)


# t = np.arange(0, 3.0, 1.0/sampling_rate)
# # wavelet transfrom 
# for i in range(len(filtered_C3_data)):
#     figureName = str(i)
#     [cwtmatr3, frequencies3] = pywt.cwt(filtered_C3_data[i], scales, wavename, 1.0/sampling_rate) 
#     [cwtmatr4, frequencies4] = pywt.cwt(filtered_C4_data[i], scales, wavename, 1.0/sampling_rate) 
#     cwtmatr = np.concatenate([abs(cwtmatr3[7:30,:]), abs(cwtmatr4[7:30,:])],axis=0)   # the sequence is C3 then C4
#     fig = plt.figure()
#     plt.contourf(cwtmatr)
#     plt.xticks([])  # remove x
#     plt.yticks([])  # remove y
#     plt.axis('off') # remove axis
#     fig.set_size_inches(800/100.0,600/100.0)#  set pixels width*height
#     plt.gca().xaxis.set_major_locator(plt.NullLocator())
#     plt.gca().yaxis.set_major_locator(plt.NullLocator())
#     plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace =0, wspace =0) 
#     plt.margins(0,0)
#     #plt.savefig(path)
#     if labels[i] == 769:
#         filepath = f'wavelet_feature\morl_LH\{i}.jpg'
#     if labels[i] == 770:
#         filepath = f'wavelet_feature\morl_RH\{i}.jpg'
#     fig.savefig(filepath)
#     fig.clear()
#     plt.close(fig)
    
# print('wavelet transfrom completed')


