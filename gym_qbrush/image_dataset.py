import glob

import numpy as np
from keras import backend as K
from keras.preprocessing.image import img_to_array, load_img

#from .image_utils import save_image_array_grid


class ImageDataset(object):
    def __init__(self, source_glob, preprocessors=[]):
        self.source_glob = source_glob
        self.preprocessors = preprocessors
        self.image_data = None
        self.load_all()

    def load_all(self):
        filenames = glob.glob(self.source_glob)
        num_images = len(filenames)
        sample = img_to_array(self.preprocess_image(load_img(filenames[0])))
        self.image_data = np.zeros((num_images,) + sample.shape).astype(np.float32)
        for file_i, filename in enumerate(filenames):
            self.image_data[file_i, :] = img_to_array(
                self.preprocess_image(load_img(filename))
            )

    def get_batch(self, batch_size):
        indexes = np.random.randint(0, self.num_images, (batch_size,))
        return self.image_data[indexes]

    def preprocess_image(self, image):
        for preprocessor in self.preprocessors:
            image = preprocessor(image)
        return image

    @property
    def num_images(self):
        return self.image_data.shape[0]

    @property
    def image_shape(self):
        return self.image_data.shape[1:]

    @property
    def num_channels(self):
        axis = -1
        if K.image_dim_ordering() == 'th':
            axis = 1
        return self.image_shape[axis]

    # def save_grid(self, filename, items=16):
    #     save_image_array_grid(self.get_batch(items), filename)
