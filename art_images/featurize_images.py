import os
import numpy as np
import h5py
from tqdm import tqdm_notebook
from keras.preprocessing import image
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.models import Model


# Download ResNet
base_model = ResNet50(weights='imagenet')

# Create a function to pull out the features
model = Model(inputs=base_model.input,
              outputs=base_model.get_layer('avg_pool').output)

# Enumerate all of the WGA images
img_folder = './images/'
img_names = [img_name for img_name in os.listdir(img_folder)]
img_names.remove('.gitignore')
img_paths = [img_folder + img_name for img_name in img_names]

# Pull out the ResNet features for each image
features = np.empty((0, model.output_shape[1]))
for img_path in tqdm_notebook(img_paths):
    img = image.load_img(img_path, target_size=(224, 224))
    array = image.img_to_array(img)
    array_expanded = np.expand_dims(array, axis=0)
    preprocessed_input = preprocess_input(array_expanded)
    features_ = model.predict(preprocessed_input)
    features = np.append(features, features_, axis=0)

# Save features
h5f = h5py.File('features.h5', 'w')
foo = h5f.create_dataset('ResNet_features', data=features)

# Save the keys
with open('image_mapping.txt', 'w') as file_handle:
    for img_name in img_names:
        file_handle.write("%s\n" % img_name)
