import os
import sys
import copy
import pickle
import pandas as pd
import h5py
import numpy as np
from scipy import spatial
from dragonfly import maximise_function
from style_transfer import transfer_style


# Input arguments
seed_image = sys.argv[1]
attribute = sys.argv[2]
# Default capital is 50
try:
    max_capital = sys.argv[3]
except IndexError:
    max_capital = 50
# Default GPU is number 0
try:
    gpu_no = sys.argv[4]
except IndexError:
    gpu_no = 0
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_no)

# Load keras after we set the GPU
from keras.models import load_model
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator


# Data and results folders
experiment_folder = ('./experimental_results/%s_%s/'
                     % (attribute, seed_image.split('/')[-1].split('.')[0]))
try:
    os.mkdir(experiment_folder)
except OSError:
    pass
art_folder = './art_images/'

# Load the classifier
classifier = load_model('./WikiArt-Emotions/resnet50.h5')

# Figure out all the responses we're dealing with so we can make an attribute labeling map
df = pd.read_pickle('./WikiArt-Emotions/data.pkl')
responses = list(df.columns)
responses.remove('Image URL')
responses.sort()
attr_map = {response: i for i, response in enumerate(responses)}


def calc_attribute_match(encoded_feature_vector, attribute):
    # Perform style transfer
    update_style(encoded_feature_vector)

    # Use ResNet to classify the current image after style transfer
    global current_image
    probs = predict_adj_matches(current_image)

    # Return the match between the current image and the target attribute
    match = obj_fun(probs, attribute)
    return match


def update_style(encoded_feature_vector):
    # Increment the counter for specifying which folder to save the images in
    global updates
    results_folder = experiment_folder + 'style_transfer_%03d' % updates
    updates += 1
    # Make the directory to store the style transfer progression (EAFP)
    try:
        os.mkdir(results_folder)
    except OSError:
        pass

    # Fetch various information that we'll be using
    global current_image

    # Find the closest image to the vector provided, then transfer its style
    global tree
    global wga_names
    _, image_index = tree.query(encoded_feature_vector)
    style_image = wga_names[image_index]
    style_path = art_folder + 'images/' + style_image
    current_image = transfer_style(current_image, style_path,
                                   result_folder=results_folder,
                                   show=False)

    # Remove the style image from the list of candidates (to avoid redos)
    del wga_names[image_index]
    global encoded_features
    encoded_features = np.delete(encoded_features, image_index, axis=0)

    # Update some global variables
    global style_images
    style_images.append(style_image)
    tree = spatial.KDTree(encoded_features)


def predict_adj_matches(img_path):
    # Pull the image
    img = image.load_img(img_path, target_size=(128, 128))

    # Standardize/preprocess the image
    array = image.img_to_array(img)
    array -= np.mean(array, axis=2, keepdims=True)
    array /= (np.std(array, axis=2, keepdims=True) + 1e-7)
    array_expanded = np.expand_dims(array, axis=0)

    # Feed the image to our trained ResNet model
    probabilities = classifier.predict(array_expanded)[0, :]
    return probabilities


def obj_fun(probs, attribute):
    attr_index = attr_map[attribute]
    match = probs[attr_index]
    return match


# Load image names and initialize the list of style images
with open(art_folder + 'image_mapping.txt', 'r') as file_handle:
    wga_names = file_handle.readlines()
wga_names = [name.split('\n')[0] for name in wga_names]

# Read the encoded features, then make a KDTree from them for matching
h5f = h5py.File(art_folder + 'features.h5', 'r')
encoded_features = h5f['encoded_features'][:]
h5f.close()
sys.setrecursionlimit(10000)  # So that we can actually build the tree
tree = spatial.KDTree(encoded_features)

# Figure out the domain for dragonfly to use
maxes = encoded_features.max(axis=0)
mins = encoded_features.min(axis=0)
domain = [(min_, max_) for min_, max_ in zip(mins, maxes)]


# Let's go!
current_image = copy.deepcopy(seed_image)
updates = 0
style_images = []
max_val, max_pt, history = maximise_function(lambda features: calc_attribute_match(features, attribute),
                                             domain, max_capital)

# Save it before we lose it!
seed_name = seed_image.split('.')[0]
with open(experiment_folder + 'history.pkl', 'wb') as file_handle:
    pickle.dump((max_val, history, style_images), file_handle)
