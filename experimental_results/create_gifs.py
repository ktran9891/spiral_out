import os
import sys
import imageio


# Input arguments
attr = sys.argv[1]
seed = sys.argv[2]
try:
    fps = sys.argv[3]
except IndexError:
    fps = 20


# Find each style transfer iteration
folder = attr + '_' + seed.split('.')[0]
subfolders = [os.path.join(folder, subfolder)
              for subfolder in os.listdir(folder)
              if '.' not in subfolder]
subfolders.sort()

# Find each image
file_paths = [os.path.join(subfolder, file_path)
              for subfolder in subfolders
              for file_path in os.listdir(subfolder)]
file_paths.sort()

# Start the GIF with the native image
images = [imageio.imread(seed) for _ in range(int(fps))]
# Add the style transfer images to the GIF
images.extend([imageio.imread(path) for path in file_paths])
imageio.mimsave(folder + '.gif', images, fps=fps)
