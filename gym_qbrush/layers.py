from keras import backend as K
from keras.layers import Layer


class ChannelFilter(Layer):
    def __init__(self, channels, *args, **kwargs):
        self.channels = channels
        super(ChannelFilter, self).__init__(*args, **kwargs)

    def call(self, x, mask=None):
        if K.image_dim_ordering() == 'tf':
            return x[:, :, :, self.channels]
        else:
            return x[:, self.channels, :, :]

    def get_output_shape_for(self, input_shape):
        if K.image_dim_ordering() == 'tf':
            return input_shape[:-1] + (len(self.channels),)
        else:
            return (input_shape[0], len(self.channels)) + input_shape[2:]
