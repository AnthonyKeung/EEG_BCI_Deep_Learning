# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import numpy as np
import matplotlib.pyplot as plt

import mne

# Path to the .gdf file
file_path = "BCI_IV_2b/B0101T.gdf"

# Load the GDF file
raw = mne.io.read_raw_gdf(file_path, verbose=True)
print("Information",raw.info)
print("Channel Names:",raw.ch_names)
raw.plot()
plt.show()