import logging
import os

import gym
import numpy as np
from gym import error, spaces
from gym import utils
from gym.utils import seeding
from keras import backend as K
from keras.applications import vgg16
from keras.layers import Input
from keras.models import Model
from keras.preprocessing.image import img_to_array
from PIL import Image, ImageDraw

from gym_qbrush import image_preprocessors
from gym_qbrush.image_dataset import ImageDataset
from gym_qbrush.np_objectives import mse


logger = logging.getLogger(__name__)


class QBrushEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}
    action_names = ['up', 'right', 'down', 'left', 'stop']

    def __init__(
            self, canvas_size=(64, 64), canvas_channels=3, color='white',
            background_color='black', position_map_decay=0.5,
            feature_layers=['block4_conv1'], move_size=0.05):
        super(QBrushEnv, self).__init__()
        self.viewer = None
        self.image_dataset = None
        self.canvas_size = canvas_size
        self.canvas_channels = canvas_channels
        self.color = color
        self.background_color = background_color
        self.position_map_decay = position_map_decay
        self.feature_layers = feature_layers
        self.move_size = move_size
        self.action_space = spaces.Discrete(len(self.action_names))
        self.observation_space = spaces.Box(
            low=0, high=255, shape=self.observation_shape)
        self._prepare_vgg()
        self._update_baseline_image()

    def set_image_source(self, glob, preprocessors=[]):
        preprocessors.append(image_preprocessors.resize(self.canvas_size))
        self.image_dataset = ImageDataset(glob, preprocessors=preprocessors)

    def _reset(self):
        self.update_target(self.image_dataset.get_batch(1)[0])
        self.canvas = self.blank_canvas()
        self._update_canvas_array()
        self.position = np.random.uniform(0., 1., (2,))
        self.position_map = np.zeros(self.canvas_shape_size)
        self._update_position_map()
        return np.array(self.canvas_arr)

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return
        if mode == 'rgb_array':
            return self.rgb_canvas.astype(np.uint8)
        elif mode == 'human':
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(self.rgb_canvas.astype(np.uint8))

    def _step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        # dispatch action
        action_name = self.action_names[action]
        #print action_name
        if action_name == 'stop':
            done = True
        else:
            done = False
            getattr(self, 'perform_move_{}'.format(action_name))()
            self._update_canvas_array()
            #print self.canvas_arr.min(), self.canvas_arr.mean(), self.canvas_arr.max()
        return self.get_state(), self.calculate_reward(done), done, {}

    def perform_move_up(self):
        self._perform_move(0., -self.move_size)

    def perform_move_down(self):
        self._perform_move(0., self.move_size)

    def perform_move_left(self):
        self._perform_move(-self.move_size, 0.)

    def perform_move_right(self):
        self._perform_move(self.move_size, 0.)

    def _perform_move(self, dx, dy):
        start = np.copy(self.position)
        canvas = self.canvas
        self.position[0] += dx
        self.position[1] += dy
        self.position = self.position.clip(0.0, 1.0)
        image_size = np.array(canvas.size) - 1
        draw = ImageDraw.Draw(canvas)
        draw.line(
            [tuple(start * image_size), tuple(self.position * image_size)],
            self.color)
        self._update_position_map()

    def get_state(self):
        if K.image_dim_ordering() == 'tf':
            axis = -1
            pmap = self.position_map[..., None]
        else:
            axis = 1
            pmap = self.position_map[None, ...]
        return np.concatenate([self.canvas_arr, self.target_arr, pmap * 255.], axis=axis)

    def blank_canvas(self):
        return Image.new(
            self.canvas_mode, self.canvas_size, self.background_color)

    def _update_canvas_array(self):
        self.canvas_arr = img_to_array(self.canvas)

    def _update_position_map(self):
        self.position_map *= self.position_map_decay
        grid_shape = np.array(self.position_map.shape) - 1
        indexes = (self.position[::-1] * grid_shape).astype(np.int32)
        self.position_map[indexes] = 1.

    def update_target(self, target_image):
        self.target_image = target_image
        self.target_arr = img_to_array(self.target_image)
        self.target_features = self.get_image_features(self.target_arr[None, ...])[0]
        self.baseline_error = mse(self.baseline_features, self.target_features)

    def _update_baseline_image(self):
        self.baseline_canvas = np.random.uniform(0., 255., self.canvas_shape)
        self.baseline_features = self.get_image_features(self.baseline_canvas[None, ...])

    def _prepare_vgg(self):
        vgg = vgg16.VGG16(include_top=False, input_shape=self.vgg_shape)
        outputs = []
        for layer_name in self.feature_layers:
            layer = vgg.get_layer(layer_name)
            outputs.append(layer.output)
        self.vgg_features = Model(vgg.inputs, outputs)
        self.vgg_features.compile(optimizer='adam', loss='mse')

    def get_image_features(self, images):
        images = rgb_array(images)
        images = vgg16.preprocess_input(images)
        return self.vgg_features.predict(images)

    @property
    def canvas_mode(self):
        return {1: 'L', 3: 'RGB'}[self.canvas_channels]

    @property
    def canvas_shape(self):
        if K.image_dim_ordering() == 'tf':
            return self.canvas_shape_size + (self.canvas_channels,)
        else:
            return (self.canvas_channels,) + self.canvas_shape_size

    @property
    def vgg_shape(self):
        if K.image_dim_ordering() == 'tf':
            return self.canvas_shape_size + (3,)
        else:
            return (3,) + self.canvas_shape_size

    @property
    def canvas_shape_size(self):
        '''The spatial dimensions of the canvas, in array order.'''
        return tuple(reversed(self.canvas_size))

    @property
    def rgb_canvas(self):
        return rgb_array(self.canvas_arr[None, ...])[0]

    @property
    def observation_shape(self):
        obs_shape = list(self.canvas_shape)
        if K.image_dim_ordering() == 'tf':
            axis = -1
        else:
            axis = 1
        obs_shape[axis] *= 2  # target image channels
        obs_shape[axis] += 1  # position map
        return obs_shape


def rgb_array(image_arr):
    '''ensure each image has 3 channels'''
    if K.image_dim_ordering() == 'tf':
        axis = -1
    else:
        axis = 1
    if image_arr.shape[axis] == 1:
        image_arr = np.repeat(image_arr, 3, axis=axis)
    return image_arr
