import os
import numpy as np
import h5py
from tqdm import tqdm
from keras.models import Model, load_model
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

# Download ResNet
base_model = load_model('../sentiment_classification/resnet50_vso.h5')

# Create a function to pull out the features from ResNet
model = Model(inputs=base_model.input,
              outputs=base_model.get_layer('flatten_1').output)

# Enumerate all of the WGA images
img_folder = './images/'
img_names = [img_name for img_name in os.listdir(img_folder)]

# Ignore some irrelevant and buggy images
img_names.remove('.gitignore')
img_names.remove('04620.jpg')
img_names.remove('14889.jpg')  
img_names.remove('11243.jpg')

# Concatenate folder in front of file names
img_paths = [img_folder + img_name for img_name in img_names]

# Get each WGA image
features = []
for img_path in tqdm(img_paths):
    img = image.load_img(img_path, target_size=(128, 128))
    array = image.img_to_array(img)

    # Standardize all the WGA images
    array -= np.mean(array, axis=2, keepdims=True)
    array /= (np.std(array, axis=2, keepdims=True) + 1e-7)

    # Feed the images to our trained ResNet model to get our features
    array_expanded = np.expand_dims(array, axis=0)
    features_ = model.predict(array_expanded)
    features.append(features_)
features = np.array(features)

# Save features
h5f = h5py.File('features.h5', 'w')
h5f.create_dataset('ResNet_features', data=features)
h5f.close()

# Save the keys
with open('image_mapping.txt', 'w') as file_handle:
    for img_name in img_names:
        file_handle.write("%s\n" % img_name)
