import os
import sys
import pickle
import numpy as np
import imageio
from PIL import Image


# Input arguments
attr = sys.argv[1]
seed = sys.argv[2]
try:
    fps = sys.argv[3]
except IndexError:
    fps = 20
size = (1024, 1024)


# Find the style images
folder = attr + '_' + seed.split('.')[0]
with open(folder + '/history.pkl', 'rb') as file_handle:
    _, _, style_images = pickle.load(file_handle)

# Find each image
art_folder = '../art_images/images/'
file_paths = [os.path.join(art_folder, file_name)
              for file_name in style_images]

# Add the experience images to the GIF
images = []
for path in file_paths:
    # Read each experience image
    image = Image.open(path)
    image.thumbnail(size, Image.ANTIALIAS)

    # Center and pad the image
    offset_x = max((size[0] - image.size[0]) / 2, 0)
    offset_y = max((size[1] - image.size[1]) / 2, 0)
    offsets = (offset_x, offset_y)
    thumbnail = Image.new(mode='RGB', size=size, color=(1, 1, 1))
    thumbnail.paste(image, offsets)

    # Save a full set of frames per image
    for _ in range(int(fps)):
        images.append(thumbnail)
imageio.mimsave(folder + '_experiences.gif', images, fps=fps)
