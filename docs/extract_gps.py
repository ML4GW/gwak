import os
import numpy as np

def extract_gps_numbers(folder_path):
    gps_numbers = []

    # Loop through all files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".png"):
            # Extract GPS number before the first "_"
            gps_number_str = filename.split("_")[0]
            gps_number = float(gps_number_str)  # Convert to float
            gps_numbers.append(gps_number)

    # Convert list to numpy array
    gps_array = np.array(gps_numbers)
    return gps_array

# Usage
folder_path_a = "all_O3a_spectrogram_boom/"  # Replace with your folder path
folder_path_b = "all_O3b_spectrogram_new/"  # Replace with your folder path
gps_numbers_a = extract_gps_numbers(folder_path_a)
gps_numbers_b = extract_gps_numbers(folder_path_b)
gps_numbers = np.concatenate((gps_numbers_a, gps_numbers_b))
print([f'{i:0.4f}' for i in gps_numbers])
np.save('gps.npy', gps_numbers)