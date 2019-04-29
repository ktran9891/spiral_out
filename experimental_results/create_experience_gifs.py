import os
import sys
import pickle
import numpy as np
import imageio


# Input arguments
attr = sys.argv[1]
seed = sys.argv[2]
try:
    fps = sys.argv[3]
except IndexError:
    fps = 20


# Find the style images
folder = attr + '_' + seed.split('.')[0]
with open(folder + '/history.pkl', 'rb') as file_handle:
    _, _, style_images = pickle.load(file_handle)

# Find each image
art_folder = '../art_images/images/'
file_paths = [os.path.join(art_folder, file_name)
              for file_name in style_images]

# Start the GIF with a black image to go along with the OG portrait
#shape = imageio.imread(seed).shape
#black = np.zeros(shape, dtype=np.uint8)
#images = [black for _ in range(int(fps))]
images = []

# Add the experienc images to the GIF
images.extend([imageio.imread(path)
               for path in file_paths
               for _ in range(int(fps))])
imageio.mimsave(folder + '_experiences.gif', images, fps=fps)
