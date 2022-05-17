import os
data_paths = [os.path.join(pth, f) for pth, dirs, files in os.walk('./2688_test') for f in files]
