import argparse
import os
from collections import deque

import gym
import gym.wrappers
import numpy as np
from keras import backend as K
from keras.layers import (
    Activation, Convolution2D, Dense, Dropout, Flatten, GlobalAveragePooling2D,
    Input, LeakyReLU, MaxPooling2D, merge, RepeatVector
)
from keras.models import Model, Sequential
from keras.optimizers import Adam, RMSprop

from yarp.advantage import AdvantageAggregator
from yarp.agent import QAgent
from yarp.memory import Memory
from yarp.policy import AnnealedGreedyQPolicy, EpsilonGreedyQPolicy

import gym_qbrush
from gym_qbrush import image_preprocessors


class QBrushAdvantageAgent(QAgent):
    def build_model(self):
        obs_space = self.environment.observation_space
        x = input = Input(shape=obs_space.low.shape)
        x = Convolution2D(32, 8, 8, subsample=(4, 4), name='conv0')(x)
        x = LeakyReLU(name='conv0_act')(x)
        x = Convolution2D(64, 4, 4, subsample=(2, 2), name='conv1')(x)
        x = LeakyReLU(name='conv1_act')(x)
        x = Convolution2D(64, 2, 2, subsample=(1, 1), name='conv2')(x)
        x = LeakyReLU(name='conv2_act')(x)
        x = Flatten()(x)
        v_hat = Dense(self.config.num_hidden)(x)
        v_hat = LeakyReLU()(v_hat)
        v_hat = Dense(1, name='v_hat')(v_hat)
        a_hat = Dense(self.config.num_hidden)(x)
        a_hat = LeakyReLU()(a_hat)
        a_hat = Dense(self.environment.action_space.n, name='a_hat')(a_hat)
        x = AdvantageAggregator(name='q')([v_hat, a_hat])
        model = Model([input], [x])
        optimizer = RMSprop(lr=self.config.lr, rho=0.99, clipnorm=10., epsilon=0.01)
        model.compile(optimizer=optimizer, loss='mse')
        return model

    def model_custom_objects(self, **kwargs):
        return super(QBrushAdvantageAgent, self).model_custom_objects(
            AdvantageAggregator=AdvantageAggregator, **kwargs)


AGENT_REGISTRY = dict(
    duel=QBrushAdvantageAgent
)

def print_stats(label, arr):
    if arr:
        print('{} // min: {} mean: {} max: {}'.format(
            label, np.min(arr), np.mean(arr), np.max(arr)
        ))


def main(config, api_key):
    print('loading images')
    print('creating environment')
    environment = gym.make(config.env)
    environment.set_image_source(
        config.images_glob, preprocessors=[image_preprocessors.greyscale])
    environment = gym.wrappers.Monitor(
        environment, config.monitor_path, force=True
    )
    environment.reset()

    print('creating agent')
    agent_class = AGENT_REGISTRY[config.agent]
    memory = Memory(config.memory)
    agent = agent_class(
        config, environment, memory, name=config.model_name,
        ignore_existing=config.ignore_existing,
    )
    train_policy = AnnealedGreedyQPolicy(
        agent, config.epsilon, config.min_epsilon,
        config.anneal_steps
    )
    eval_policy = EpsilonGreedyQPolicy(agent, 0.01)
    print('simulating...')
    epsilon = config.epsilon
    d_epsilon = 1. / float(config.anneal_steps) * config.epsilon
    needs_training = True
    recent_episode_rewards = deque([], 100)
    try:
        for epoch_i in range(config.epochs):
            print('epoch {} / epsilon = {}'.format(epoch_i, train_policy.epsilon))
            if needs_training:
                losses, rewards, episode_rewards = agent.train(
                    environment, train_policy, train_p=1.0, max_steps=config.learn_steps,
                    max_episodes=config.learn_episodes
                )
                train_policy.step(len(rewards))
                recent_episode_rewards += episode_rewards
                print_stats('Loss', losses)
                print_stats('All rewards', rewards)
                print_stats('Episode rewards', episode_rewards)
            else:
                print('skipping training...')
            # Evaluate
            losses, rewards, episode_rewards = agent.train(
                environment, eval_policy, train_p=0.,
                max_steps=config.sim_steps, max_episodes=config.sim_episodes
            )
            recent_episode_rewards += episode_rewards
            print_stats('Loss', losses)
            print_stats('All rewards', rewards)
            print_stats('Episode rewards', episode_rewards)
            if (epoch_i + 1) % config.save_rate == 0:
                agent.save_model()
    except KeyboardInterrupt:
        pass
    #environment.monitor.close()
    # if api_key:
    #     print('uploading')
    #     gym.upload('monitor-data', api_key=api_key)


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser('qbrush solver')
    arg_parser.add_argument('--discount', type=float, default=0.99)
    arg_parser.add_argument('--episodes', type=int, default=5)
    arg_parser.add_argument('--epsilon', type=float, default=1.0)
    arg_parser.add_argument('--min-epsilon', type=float, default=0.05)
    arg_parser.add_argument('--epochs', type=int, default=10000)
    arg_parser.add_argument('--anneal-steps', type=int, default=100000)
    arg_parser.add_argument('--sim-steps', type=int, default=300)
    arg_parser.add_argument('--sim-episodes', type=int, default=10)
    arg_parser.add_argument('--learn-steps', type=int, default=3000)
    arg_parser.add_argument('--learn-episodes', type=int, default=100)
    arg_parser.add_argument('--ignore-existing', action='store_true')
    arg_parser.add_argument('--model-name', default='qbrush')
    arg_parser.add_argument('--save-rate', type=int, default=10)
    arg_parser.add_argument('--num-hidden', type=int, default=128)
    arg_parser.add_argument('--memory', type=int, default=40000)
    arg_parser.add_argument('--agent', default='duel')
    arg_parser.add_argument('--lr', type=float, default=0.0000625)
    arg_parser.add_argument('--monitor-path', default='monitor-data')
    arg_parser.add_argument('--env', default='awentzonline/QBrush-Step-v0')
    arg_parser.add_argument('--target-update', type=float, default=1e-3)
    arg_parser.add_argument('images_glob')

    config = arg_parser.parse_args()

    api_key = os.environ.get('AIGYM_API_KEY' ,'').strip()
    print('api key:' + api_key)
    main(config, api_key)
