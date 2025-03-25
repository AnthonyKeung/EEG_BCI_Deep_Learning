
import numpy as np
import matplotlib.pyplot as plt 
import mne

def raw_channel_into_frames(raw_channel, eventpos):
    #[[     0      0      8]
    #  [   499      0      6]
    #  [ 16749      0      7]
    #  [ 32999      0      5]
    #  [ 37999      0      4]
    #  [ 42999      0      3]
    #  [ 47999      0      2]
    #  [ 52251      0      8]
    #  [ 55139      0      9]
    # Find the first instance of 9 in the 3rd column
    start_trial_index = np.argmax(eventpos[:, 2] == 9)

    # Iterate through the event positions
    channel_frames = np.empty((0, 750))
    rejected_trials = 0
    current_trial = start_trial_index
    while (current_trial < len(eventpos[:,0])):
        # Extract the start and end indices of the trial
        print("i:",current_trial)

        if eventpos[current_trial , 2] == 9 and eventpos[current_trial+1, 2] != 1:
            print("Trial Start:", eventpos[current_trial , 0])
            print("Trial End Index:", eventpos[current_trial+1, 0])
            print("Window size is ", eventpos[current_trial+1, 0] - eventpos[current_trial , 0])
            trial_start_index = eventpos[current_trial , 0]
            print("Trial Start:", trial_start_index)
            trial_end_index = eventpos[current_trial+1, 0] if current_trial+1 < len(eventpos) else len(raw_channel)
            print("Trial End Index:", trial_end_index) 
            print("Window size is ", trial_end_index - trial_start_index)

            # Extract the trial data
            trial_data = raw_channel[0, trial_start_index:trial_end_index]

            print("Trial Data Shape:", trial_data.shape)
            # Append the trial data to the 2D array
            channel_frames = np.vstack((channel_frames, trial_data))
            current_trial += 2
        else:
            current_trial += 3
        
    print("Last trial", current_trial)
    print("Channel Size",channel_frames.shape)
    return channel_frames



def gdf_to_raw_data_input(gdf_filepath):
    # Load the GDF file
    #(750, 3 ) 750 timepoints, 3 channels
    raw = mne.io.read_raw_gdf(gdf_filepath, verbose=False)
    eventpos, event_dur = mne.events_from_annotations(raw)
    
    channel_index = raw.ch_names.index('EEG:C3')  # Get the index of channel 'C3'
    c3_data, _ = raw[channel_index]  # Extract the data and corresponding times
    # print("C3 Data Shape:", c3_data.shape) ##C3 Data Shape: (1, 604803)
    #print("Type of C3 Data:", type(c3_data)) ##tis a numpy array
    # print("C3 Data:", c3_data) ## [[ 4.15045396e-06  4.44647898e-06  2.20035096e-06 ...  2.11795224e-06, 1.40993362e-06 -6.71396963e-07]]

    channel_index = raw.ch_names.index('EEG:Cz')  # Get the index of channel 'C3'
    cz_data, _ = raw[channel_index]  # Extract the data and corresponding times

    channel_index = raw.ch_names.index('EEG:C4')  # Get the index of channel 'C3'
    c4_data, _ = raw[channel_index]  # Extract the data and corresponding times

    raw_channel_into_frames(c3_data, eventpos)

    # # Ensure all channels have the same length
    # if not (len(c3_data[0]) == len(cz_data[0]) == len(c4_data[0])):
    #     raise ValueError("Channel data lengths are not equal. Ensure all channels have the same length.")
    
    # # Stack the channels into a 2D array (timepoints, channels)
    # data = np.stack([c3_data[0], cz_data[0], c4_data[0]], axis=1)  # Shape: (timepoints, 3)
    # print(type(data))
    # print(data.shape)
    # print(data)
    
    # return data

if __name__ == '__main__':
    file_path = "BCI_IV_2b/B0101T.gdf"
    raw_data = gdf_to_raw_data_input(file_path)

# [   499      0      6]
#  [ 16749      0      7]
#  [ 32999      0      5]
#  [ 37999      0      4]
#  [ 42999      0      3]
#  [ 47999      0      2]
#  [ 52251      0      8]
#  [ 55139      0      9]
#  [ 55889      0     10]
#  [ 57499      0      9]
#  [ 58249      0     11]
#  [ 62067      0      9]
#  [ 62817      0     10]
#  [ 64104      0      9]
#  [ 64854      0     11]
#  [ 72981      0      9]
#  [ 73731      0     11]
#  [ 75188      0      9]
#  [ 75938      0     11]
#  [ 77589      0      9]
#  [ 78339      0     11]
#  [ 88500      0      9]
#  [ 88500      0      1]
#  [ 89250      0     11]
#  [ 90934      0      9]
#  [ 91684      0     10]
#  [ 93193      0      9]
#  [ 93943      0     11]
#  [ 99953      0      9]
#  [100703      0     11]
#  [107106      0      9]
#  [107856      0     10]
#  [109567      0      9]
#  [110317      0     11]
#  [112028      0      9]
#  [112778      0     11]
#  [116391      0      9]
#  [116391      0      1]
#  [117141      0     10]
#  [121161      0      9]
#  [121911      0     10]
#  [123611      0      9]
#  [124361      0     10]
#  [132646      0      9]
#  [133396      0     10]
#  [139431      0      9]
#  [139431      0      1]
#  [140181      0     10]
#  [141477      0      9]
#  [142227      0     10]
#  [144343      0      8]
#  [144941      0      9]
#  [145691      0     10]
#  [147231      0      9]
#  [147981      0     10]
#  [149591      0      9]
#  [150341      0     10]
#  [156196      0      9]
#  [156946      0     11]
#  [160591      0      9]
#  [161341      0     10]
#  [165073      0      9]
#  [165823      0     10]
#  [169681      0      9]
#  [170431      0     10]
#  [171990      0      9]
#  [171990      0      1]
#  [172740      0     11]
#  [176248      0      9]
#  [176998      0     11]
#  [178305      0      9]
#  [179055      0     10]
#  [183026      0      9]
#  [183776      0     10]
#  [185285      0      9]
#  [186035      0     11]
#  [192045      0      9]
#  [192795      0     11]
#  [199198      0      9]
#  [199198      0      1]
#  [199948      0     11]
#  [210915      0      9]
#  [211665      0     11]
#  [215703      0      9]
#  [216453      0     10]
#  [220267      0      9]
#  [221017      0     11]
#  [222457      0      9]
#  [223207      0     11]
#  [227193      0      9]
#  [227943      0     10]
#  [229509      0      9]
#  [230259      0     11]
#  [236435      0      8]
#  [241683      0      9]
#  [242433      0     10]
#  [244005      0      9]
#  [244755      0     10]
#  [250578      0      9]
#  [251328      0     10]
#  [252683      0      9]
#  [253433      0     10]
#  [257165      0      9]
#  [257915      0     11]
#  [259372      0      9]
#  [260122      0     11]
#  [264082      0      9]
#  [264832      0     11]
#  [266134      0      9]
#  [266884      0     11]
#  [268340      0      9]
#  [269090      0     10]
#  [281770      0      9]
#  [282520      0     10]
#  [284137      0      9]
#  [284887      0     11]
#  [286576      0      9]
#  [287326      0     11]
#  [288988      0      9]
#  [289738      0     10]
#  [296212      0      9]
#  [296962      0     11]
#  [298259      0      9]
#  [298259      0      1]
#  [299009      0     11]
#  [310155      0      9]
#  [310905      0     11]
#  [314549      0      9]
#  [315299      0     10]
#  [316830      0      9]
#  [317580      0     10]
#  [319285      0      9]
#  [320035      0     11]
#  [325661      0      9]
#  [325661      0      1]
#  [326411      0     10]
#  [328527      0      8]
#  [331415      0      9]
#  [331415      0      1]
#  [332165      0     10]
#  [338343      0      9]
#  [338343      0      1]
#  [339093      0     10]
#  [342670      0      9]
#  [343420      0     11]
#  [346802      0      9]
#  [347552      0     11]
#  [353865      0      9]
#  [354615      0     10]
#  [356174      0      9]
#  [356924      0     11]
#  [358226      0      9]
#  [358976      0     11]
#  [360432      0      9]
#  [361182      0     10]
#  [362489      0      9]
#  [363239      0     10]
#  [369469      0      9]
#  [369469      0      1]
#  [370219      0     10]
#  [371753      0      9]
#  [372503      0     11]
#  [373862      0      9]
#  [373862      0      1]
#  [374612      0     11]
#  [376229      0      9]
#  [376979      0     11]
#  [378668      0      9]
#  [379418      0     11]
#  [388304      0      9]
#  [389054      0     11]
#  [397437      0      9]
#  [398187      0     10]
#  [402247      0      9]
#  [402997      0     10]
#  [411377      0      9]
#  [412127      0     10]
#  [413693      0      9]
#  [414443      0     10]
#  [417753      0      9]
#  [417753      0      1]
#  [418503      0     11]
#  [420619      0      8]
#  [423507      0      9]
#  [424257      0     10]
#  [428189      0      9]
#  [428939      0     10]
#  [430435      0      9]
#  [431185      0     11]
#  [434762      0      9]
#  [435512      0     10]
#  [436867      0      9]
#  [436867      0      1]
#  [437617      0     11]
#  [441349      0      9]
#  [442099      0     11]
#  [445957      0      9]
#  [446707      0     10]
#  [448266      0      9]
#  [449016      0     10]
#  [459302      0      9]
#  [460052      0     11]
#  [463845      0      9]
#  [464595      0     11]
#  [468321      0      9]
#  [469071      0     11]
#  [477935      0      9]
#  [477935      0      1]
#  [478685      0     11]
#  [480396      0      9]
#  [481146      0     11]
#  [482443      0      9]
#  [483193      0     10]
#  [484759      0      9]
#  [485509      0     10]
#  [489529      0      9]
#  [490279      0     10]
#  [491979      0      9]
#  [492729      0     11]
#  [501014      0      9]
#  [501764      0     10]
#  [503469      0      9]
#  [504219      0     11]
#  [509845      0      9]
#  [509845      0      1]
#  [510595      0     10]
#  [512711      0      8]
#  [517959      0      9]
#  [517959      0      1]
#  [518709      0     10]
#  [526854      0      9]
#  [527604      0     11]
#  [528959      0      9]
#  [528959      0      1]
#  [529709      0     11]
#  [530986      0      9]
#  [531736      0     11]
#  [533441      0      9]
#  [534191      0     10]
#  [540358      0      9]
#  [541108      0     10]
#  [551394      0      9]
#  [552144      0     11]
#  [553653      0      9]
#  [554403      0     10]
#  [558046      0      9]
#  [558046      0      1]
#  [558796      0     10]
#  [560413      0      9]
#  [561163      0     10]
#  [568316      0     11]
#  [572488      0      9]
#  [573238      0     10]
#  [574535      0      9]
#  [575285      0     11]
#  [579283      0      9]
#  [580033      0     10]
#  [581621      0      9]
#  [582371      0     11]
#  [584071      0      9]
#  [584821      0     10]
#  [588635      0      9]
#  [589385      0     11]
#  [590825      0      9]
#  [591575      0     10]
#  [593106      0      9]
#  [593856      0     11]
#  [601937      0      9]
#  [602687      0     11]]