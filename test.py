import pickle
import numpy as np

# Load existing calibration data
filename = './params/calibration_data_camera3_2.pkl'

with open(filename, 'rb') as f:
    calibration_data = pickle.load(f)

# Check existing keys (for reference)
print("Before adding extrinsic_matrix:", calibration_data.keys())

# Add extrinsic_matrix to the calibration data
K = calibration_data['K'] 
K = K/2
K[2][2] = 1
calibration_data['K'] = K

# Save the updated data back to the file
with open(filename, 'wb') as f:
    pickle.dump(calibration_data, f)

print("Extrinsic matrix added successfully.")