import time
from scipy.misc import imsave
from scipy.optimize import fmin_l_bfgs_b
from .model_compiler import compile_model
from .loss_functions import LossEvaluator
from .image_processing import deprocess_image, show_image


def transfer_style(content_file_name, style_file_name, iterations=20,
                   content_weight=0.025, total_variation_weight=1.0,
                   style_weight=1.0, content_layer= 'block5_conv2',
                   style_layers=None, result_folder='./images'):
    if style_layers is None:
        style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1',
                        'block4_conv1', 'block5_conv1']

    # build the VGG19 network with our 3 images as input
    # the model will be loaded with pre-trained ImageNet weights
    model = compile_model(content_file_name, style_image_path)
    evaluator = LossEvaluator(model, content_layer=content_layer, style_layers=style_layers)

    # minimise the loss function
    for i in range(iterations):
        print('Start of iteration', i)
        start_time = time.time()
        x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x.flatten(),
                                         fprime=evaluator.grads, maxfun=20)
        print('Current loss value:  ', min_val)

        # save current generated image
        img = deprocess_image(x.copy())
        fname = result_folder + '/%03d.png' % i
        imsave(fname, img)
        print('Image saved as %s', fname)

        # Report total run time
        end_time = time.time()
        print('Iteration %d completed in %ds' % (i, end_time - start_time))

    show_image(fname)
