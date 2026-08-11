import numpy as np

data = np.load("calibration/calibration_data.npz")

print("Keys in calibration file:")
print(data.files)

for key in data.files:
    print(f"\n{key}")
    print(data[key])
