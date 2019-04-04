import time
from scipy.misc import imsave
from scipy.optimize import fmin_l_bfgs_b
from keras.applications import vgg19
from .loss_functions import LossEvaluator
from .image_processing import (parse_images,
                               preprocess_image,
                               deprocess_image,
                               show_image)


def transfer_style(content_file_name, style_file_name, iterations=20,
                   content_weight=0.025, total_variation_weight=1.0,
                   style_weight=1.0, content_layer= 'block5_conv2',
                   style_layers=None, result_folder='./images',
                   img_nrows=400):
    if style_layers is None:
        style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1',
                        'block4_conv1', 'block5_conv1']

    # build the VGG19 network with our 3 images as input
    # the model will be loaded with pre-trained ImageNet weights
    model, combo_image, img_dims = compile_model(content_file_name,
                                                 style_file_name,
                                                 img_nrows=img_nrows)
    evaluator = LossEvaluator(model, combo_image=combo_image,
                              content_weight=content_weight,
                              style_weight=style_weight,
                              total_variation_weight=total_variation_weight,
                              content_layer=content_layer,
                              style_layers=style_layers,
                              img_dims=img_dims)

    # minimise the loss function
    x = preprocess_image(content_file_name, img_dims)
    for i in range(iterations):
        print('Start of iteration %i' % i)
        start_time = time.time()
        x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x.flatten(),
                                         fprime=evaluator.grads, maxfun=20)
        print('Current loss value:  %f' % min_val)

        # save current generated image
        img = deprocess_image(x.copy(), img_dims)
        fname = result_folder + '/%03d.png' % i
        imsave(fname, img)
        print('Image saved as %s' % fname)

        # Report total run time
        end_time = time.time()
        print('Iteration %d completed in %ds' % (i, end_time - start_time))

    show_image(fname)
    return fname


def compile_model(content_file_name, style_file_name, img_nrows):
    # Display the images for checking
    show_image(content_file_name)
    show_image(style_file_name)

    # Parse the images into tensor, then feed the tensor into vgg19
    input_tensor, combo_image, img_dims = parse_images(content_file_name, style_file_name, img_nrows)
    model = vgg19.VGG19(input_tensor=input_tensor, weights='imagenet', include_top=False)
    return model, combo_image, img_dims
