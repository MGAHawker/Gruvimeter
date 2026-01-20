# -*- coding: utf-8 -*-
"""
Created on Tue Jul 29 22:44:10 2025

@author: migsh
"""
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import optoanalysis
import glob
import os
import importlib
import importlib_metadata as metadata
importlib.metadata = metadata

# Get all the .trc files, manual loading would be ridiculously long
base_dir = "D:/New folder/230725/06102025-8Vpp/"
file_pattern = os.path.join(base_dir, "C1--Trace--*.trc")
file_paths = sorted(glob.glob(file_pattern),
                   key=lambda x: int(x.split('--')[-1].split('.')[0]))

# I want to save the produced timetraces for later reference,
# this is where they're going
local_output_dir = "D:/Timetraces/06Oct-24Oct set"

# Load trc files from C1(detector 1) or C2(detector 2)
# Load all the data
for file_path in file_paths:
    
    # Load data from files collected above
    data = optoanalysis.load_data(file_path)
    #How much time data is shown? Symmetric loading still happening, faulty install?
    data.plot_time_data(timeStart=-600, timeEnd=600, units='s')
    
    # Extract filename for naming the saved plot
    base_filename = os.path.splitext(os.path.basename(file_path))[0]
        
    # Save the current matplotlib figure
    save_path = os.path.join(local_output_dir, f"{base_filename}_plot.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot locally: {save_path}")
    
    # Close the current figure to free memory
    plt.close()