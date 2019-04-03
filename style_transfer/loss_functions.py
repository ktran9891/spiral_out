from keras import backend as K


class LossEvaluator(object):
    '''
    this Evaluator class makes it possible
    to compute loss and gradients in one pass
    while retrieving them via two separate functions,
    "loss" and "grads". This is done because scipy.optimize
    requires separate functions for loss and gradients,
    but computing them separately would be inefficient.
    '''
    def __init__(self, model, content_layer='block5_conv2', style_layers=None):
        if style_layers is None:
            style_layers = ['block1_conv1', 'block2_conv1',
                              'block3_conv1', 'block4_conv1',
                              'block5_conv1']

        # User-specificed attributes
        self.content_layer = content_layer
        self.style_layers = style_layers
        self.model = model

        # Set attributes
        self.loss_value = None
        self.grads_values = None

        # Calculated attributes
        self.loss = self.define_loss()
        self.f_outputs = self.get_gradients()


    def define_loss_tensor(self):
        # get the symbolic outputs of each "key" layer (we gave them unique names).
        outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])

        # compute the neural style loss
        loss = K.variable(0.)
        layer_features = outputs_dict[self.content_layer]
        base_image_features = layer_features[0, :, :, :]
        combination_features = layer_features[2, :, :, :]
        loss += content_weight * content_loss(base_image_features,
                                              combination_features)

        for layer_name in self.style_layers:
            layer_features = outputs_dict[layer_name]
            style_reference_features = layer_features[1, :, :, :]
            combination_features = layer_features[2, :, :, :]
            sl = style_loss(style_reference_features, combination_features)
            loss += (style_weight / len(self.style_layers)) * sl
        loss += total_variation_weight * total_variation_loss(combination_image)
        self.loss_tensor = loss


    def get_gradients(self):
        # get the gradients of the generated image wrt the loss
        grads = K.gradients(self.loss_tensor, combination_image)
        outputs = [self.loss_tensor]
        if isinstance(grads, (list, tuple)):
            outputs += grads
        else:
            outputs.append(grads)
        f_outputs = K.function([combination_image], outputs)
        self.f_outputs = f_outputs


    def eval_loss_and_grads(self, x):
        if K.image_data_format() == 'channels_first':
            x = x.reshape((1, 3, img_nrows, img_ncols))
        else:
            x = x.reshape((1, img_nrows, img_ncols, 3))

        outs = self.f_outputs([x])
        loss_value = outs[0]
        if len(outs[1:]) == 1:
            grad_values = outs[1].flatten().astype('float64')
        else:
            grad_values = np.array(outs[1:]).flatten().astype('float64')
        return loss_value, grad_values


    def loss(self, x):
        assert self.loss_value is None
        loss_value, grad_values = self.eval_loss_and_grads(x)
        self.loss_value = loss_value
        self.grad_values = grad_values
        return self.loss_value


    def grads(self, x):
        assert self.loss_value is not None
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        return grad_values


def gram_matrix(x):
    '''
    the gram matrix of an image tensor (feature-wise outer product)
    '''
    assert K.ndim(x) == 3
    if K.image_data_format() == 'channels_first':
        features = K.batch_flatten(x)
    else:
        features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    gram = K.dot(features, K.transpose(features))
    return gram

    
def style_loss(style, combination):
    '''
    the "style loss" is designed to maintain
    the style of the reference image in the generated image.
    It is based on the gram matrices (which capture style) of
    feature maps from the style reference image
    and from the generated image
    '''
    assert K.ndim(style) == 3
    assert K.ndim(combination) == 3
    
    S = gram_matrix(style)
    C = gram_matrix(combination)
    channels = 3
    size = img_nrows * img_ncols
    return K.sum(K.square(S - C)) / (4. * (channels ** 2) * (size ** 2))
    

def content_loss(base, combination):
    '''
    an auxiliary loss function
    designed to maintain the "content" of the
    base image in the generated image
    '''
    return K.sum(K.square(combination - base))


def total_variation_loss(x):
    '''
    the 3rd loss function, total variation loss,
    designed to keep the generated image locally coherent (no big changes)
    '''
    assert K.ndim(x) == 4
    if K.image_data_format() == 'channels_first':
        a = K.square(x[:, :, :img_nrows - 1, :img_ncols - 1] - x[:, :, 1:, :img_ncols - 1])
        b = K.square(x[:, :, :img_nrows - 1, :img_ncols - 1] - x[:, :, :img_nrows - 1, 1:])
    else:
        a = K.square(x[:, :img_nrows - 1, :img_ncols - 1, :] - x[:, 1:, :img_ncols - 1, :])
        b = K.square(x[:, :img_nrows - 1, :img_ncols - 1, :] - x[:, :img_nrows - 1, 1:, :])
    return K.sum(K.pow(a + b, 1.25))
