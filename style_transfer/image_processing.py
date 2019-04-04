import numpy as np
from scipy.misc import imread
import matplotlib.pyplot as plt
from keras import backend as K
from keras.preprocessing.image import load_img, img_to_array
from keras.applications import vgg19


def parse_images(content_file_name, style_file_name, img_nrows):
    # get dimensions (width, height) of the generated picture
    width, height = load_img(content_file_name).size
    img_ncols = int(width * img_nrows / height)

    # get tensor representations of our images
    content_image = K.variable(preprocess_image(content_file_name,
                                                (img_nrows, img_ncols)))
    style_image = K.variable(preprocess_image(style_file_name,
                                              (img_nrows, img_ncols)))

    # this will contain our generated image
    if K.image_data_format() == 'channels_first':
        combo_image = K.placeholder((1, 3, img_nrows, img_ncols))
    else:
        combo_image = K.placeholder((1, img_nrows, img_ncols, 3))

    # Concatenate all of the image tensors into one
    input_tensor = K.concatenate([content_image, style_image, combo_image], axis=0)
    return input_tensor, combo_image, (img_nrows, img_ncols)


def preprocess_image(image_path, img_dims):
    '''
    pre-process the image: rescaling, running it through VGG19
    '''
    img_nrows, img_ncols = img_dims
    img = load_img(image_path, target_size=(img_nrows, img_ncols))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = vgg19.preprocess_input(img)
    return img


def deprocess_image(x, img_dims):
    '''
    utility function to convert a tensor into a valid image
    '''
    img_nrows, img_ncols = img_dims
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
