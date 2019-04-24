import os
import imageio
from tqdm import tqdm_notebook


# Find each ANP
folders = [folder for folder in os.listdir('.') if '.' not in folder]
for folder in tqdm_notebook(folders, desc='ANP'):

    # Find each style transfer iteration
    subfolders = [os.path.join(folder, subfolder)
                  for subfolder in os.listdir(folder)
                  if '.' not in subfolder]
    subfolders.sort()

    # Find each image
    file_paths = [os.path.join(subfolder, image_path)
                  for subfolder in subfolders
                  for file_path in os.listdir(subfolder)]
    file_paths.sort()

    # Put the images into a GIF
    images = [imageio.imread(path) for path in file_paths]
    imageio.mimsave(folder + '.gif', images, fps=20)
