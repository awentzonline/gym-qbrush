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


class StepQualityRewardEnv(QBrushEnv):
    '''Give a reward each step if the canvas error has improved.'''
    def _reset(self):
        super(StepQualityRewardEnv, self)._reset()
        self.best_canvas_err = mse(self.baseline_features, self.target_features)

    def calculate_reward(self, done):
        if done:
            return -1.
        else:
            canvas_features = self.get_image_features(self.canvas_arr[None, ...])[0]
            canvas_err = mse(canvas_features, self.target_features)
            if canvas_err < self.best_canvas_err:
                reward = 10. * self.best_canvas_err / canvas_err
                self.best_canvas_err = canvas_err
            else:
                reward = -1.
        return reward

    def _update_baseline_image(self):
        self.baseline_canvas = img_to_array(self.blank_canvas())
        self.baseline_features = self.get_image_features(self.baseline_canvas[None, ...])
