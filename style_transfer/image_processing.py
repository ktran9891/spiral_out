import numpy as np
from scipy.misc import imread
import matplotlib.pyplot as plt
from keras import backend as K
from keras.preprocessing.image import load_img, img_to_array
from keras.applications import vgg19


def preprocess_image(image_path):
    '''
    pre-process the image: rescaling, running it through VGG19
    '''
    img = load_img(image_path, target_size=(img_nrows, img_ncols))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = vgg19.preprocess_input(img)
    return img


def deprocess_image(x):
    '''
    utility function to convert a tensor into a valid image
    '''
    if K.image_data_format() == 'channels_first':
        x = x.reshape((3, img_nrows, img_ncols))
        x = x.transpose((1, 2, 0))
    else:
        x = x.reshape((img_nrows, img_ncols, 3))

    # Remove zero-center by mean pixel
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68

    # 'BGR'->'RGB'
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype('uint8')
    return x


def show_image(file_name):
    image_content = imread(file_name)
    plt.figure(figsize=(11, 11))
    plt.imshow(image_content)
