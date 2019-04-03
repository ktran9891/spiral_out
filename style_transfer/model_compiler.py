from keras import backend as K
from keras.preprocessing.image import load_img
from keras.applications import vgg19
from .image_processing import deprocess_image, show_image


def compile_model(content_file_name, style_file_name):
    show_image(content_file_name)
    show_image(style_file_name)

    # get dimensions (width, height) of the generated picture
    width, height = load_img(content_file_name).size
    img_nrows = 400 # recale the image to 400 pixel rows
    img_ncols = int(width * img_nrows / height)

    # get tensor representations of our images
    base_image = K.variable(preprocess_image(content_file_name))
    style_reference_image = K.variable(preprocess_image(style_file_name))

    # this will contain our generated image
    if K.image_data_format() == 'channels_first':
        combination_image = K.placeholder((1, 3, img_nrows, img_ncols))
    else:
        combination_image = K.placeholder((1, img_nrows, img_ncols, 3))

    # combine the 3 images (style, content, result image that starts from the
    # white noise) into a single Keras tensor
    input_tensor = K.concatenate([base_image,
                                  style_reference_image,
                                  combination_image], axis=0)

    model = vgg19.VGG19(input_tensor=input_tensor, weights='imagenet', include_top=False)
    return model
