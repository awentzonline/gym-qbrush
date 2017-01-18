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

from gym_qbrush.np_objectives import mse
from .base import QBrushEnv


logger = logging.getLogger(__name__)


class FinalQualityRewardEnv(QBrushEnv):
    '''Give one positive reward at the end proportional to the final canvas error.'''
    def calculate_reward(self, done):
        if done:
            # calculate reward relative to quality of a random image
            canvas_features = self.get_image_features(self.canvas_arr[None, ...])[0]
            canvas_err = mse(canvas_features, self.target_features)
            err_ratio = self.baseline_error / canvas_err
            reward = (err_ratio - 1.) * 10 # TODO: improve this
        else:
            reward = -0.1
        return reward
