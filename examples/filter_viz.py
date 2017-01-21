'''Visualization of the filters of the agent's CNN, via gradient ascent in input space.
This script can run on CPU in a few minutes (with the TensorFlow backend).
Results example: http://i.imgur.com/4nj4KjN.jpg

This was adapted from a Keras example.
'''
from __future__ import print_function
import sys

import numpy as np
import time
from keras.applications import vgg16
from keras import backend as K
from keras.models import load_model
from scipy.misc import imsave

from yarp.advantage import AdvantageAggregator
from gym_qbrush.layers import ChannelFilter


# dimensions of the generated pictures for each filter.
img_width = 64
img_height = 64

# the name of the layer we want to visualize
layer_name = 'leakyrelu_1'

# util function to convert a tensor into a valid image


def deprocess_image(x):
    if K.image_dim_ordering() == 'th':
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x

# build the network
print('Loading model...')
model = load_model(
    sys.argv[1], custom_objects=dict(
        AdvantageAggregator=AdvantageAggregator,
        ChannelFilter=ChannelFilter))
model.summary()

# this is the placeholder for the input images
input_img = model.input

# get the symbolic outputs of each "key" layer (we gave them unique names).
layer_dict = dict([(layer.name, layer) for layer in model.layers[1:]])

print(layer_dict)
def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)


kept_filters = []
for filter_index in range(0, 32):
    print('Processing filter %d' % filter_index)
    start_time = time.time()

    # we build a loss function that maximizes the activation
    # of the nth filter of the layer considered
    layer_output = layer_dict[layer_name].output
    if K.image_dim_ordering() == 'th':
        loss = K.mean(layer_output[:, filter_index, :, :])
    else:
        loss = K.mean(layer_output[:, :, :, filter_index])

    # we compute the gradient of the input picture wrt this loss
    grads = K.gradients(loss, input_img)[0]

    # normalization trick: we normalize the gradient
    grads = normalize(grads)

    # this function returns the loss and grads given the input picture
    iterate = K.function([input_img], [loss, grads])

    # step size for gradient ascent
    step = 1.

    # we start from a gray image with some random noise
    num_input_filters = 3  # canvas, target, position
    if K.image_dim_ordering() == 'th':
        filter_axis = 1
        input_img_data = np.random.random((1, num_input_filters, img_width, img_height))
        #input_img_data[0,-1] = 0.
    else:
        filter_axis = -1
        input_img_data = np.random.random((1, img_width, img_height, num_input_filters))
        #input_img_data[0,...,-1] = 0.
    # we run gradient ascent for 20 steps
    for i in range(20):
        loss_value, grads_value = iterate([input_img_data])
        input_img_data += grads_value * step

        print('Current loss value:', loss_value)
        if loss_value <= 0.:
            # some filters get stuck to 0, we can skip them
            break

    # decode the resulting input image
    if loss_value > 0:
        img = deprocess_image(input_img_data[0])
        kept_filters.append((img, loss_value))
    end_time = time.time()
    print('Filter %d processed in %ds' % (filter_index, end_time - start_time))

# we will stich the best 64 filters on a 8 x 8 grid.
n = 8

# the filters that have the highest loss are assumed to be better-looking.
# we will only keep the top 64 filters.
kept_filters.sort(key=lambda x: x[1], reverse=True)
kept_filters = kept_filters[:n * n]

# build a black picture with enough space for
# our 8 x 8 filters of size 128 x 128, with a 5px margin in between
margin = 5
width = n * img_width + (n - 1) * margin
height = n * img_height + (n - 1) * margin
stitched_filters = np.zeros((width, height, 3))

print(len(kept_filters))
# fill the picture with our saved filters
for i in range(n):
    for j in range(n):
        if i * n + j >= len(kept_filters):
            break
        img, loss = kept_filters[i * n + j]
        stitched_filters[(img_width + margin) * i: (img_width + margin) * i + img_width,
                         (img_height + margin) * j: (img_height + margin) * j + img_height, :] = img

# save the result to disk
imsave('output/canvas_filters_%dx%d.png' % (n, n), stitched_filters[..., 0])
imsave('output/target_filters_%dx%d.png' % (n, n), stitched_filters[..., 1])
imsave('output/position_filters_%dx%d.png' % (n, n), stitched_filters[..., 2])
