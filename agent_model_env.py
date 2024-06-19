import numpy as np
import pandas as pd

import time
import gymnasium as gym
from gymnasium import spaces
import random
import numpy as np

from glob import glob
import re

from enum import Enum

from stable_baselines3.dqn import MlpPolicy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO
from stable_baselines3.common import monitor 
from stable_baselines3.common import logger
from gymnasium.envs.registration import register

import agent_model as model

class FurnaceEnv(gym.Env):

    metadata = {"render_modes": [None], 'render_fps': 1}

    def __init__(self, train_x, train_y, row_per_episode=1):
        super().__init__()

        # Inicializar el actor
        self.furnace_actor = model.FurnaceModel(train_x, train_y)

        # Crear action space con las acciones
        self.action_space = gym.spaces.Discrete(len(model.Action))

        # Crear observation space
        self.observation_space = spaces.Box(0, 1e10, shape=(7,), dtype=np.float32)

        # Initialize parameters
        self.row_per_episode = row_per_episode
        self.step_count = 0
        self.random = random
        self.dataset_idx = 0

    def reset(self, seed=None, options={}):
        super().reset(seed=seed, options=options)

        self.furnace_actor.reset(seed=seed)
        self.step_count = 0
        info = {}
        obs = self._next_obs()

        return obs, info

    def furnace_actor_reward(self):
        if self.furnace_actor.potencia <= 370:
            return 1
        else:
            return 0

    def step(self, action):

        # Realiza acción
        self.furnace_actor.perform_action(model.Action(action))

        # Determina reward
        done = False
        reward = self.furnace_actor_reward()

        self.step_count += 1
        if self.step_count >= self.row_per_episode:
            done = True

        # Obtiene observación siguiente
        obs = self._next_obs()

        # Devuelve observación, reward, terminated, truncated
        return obs, reward, done, False, {}
    
    def _next_obs(self):

        obs = self.furnace_actor.x[self.dataset_idx]
        self.expected_action = self.furnace_actor.y[self.dataset_idx]

        self.dataset_idx += 1
        if self.dataset_idx >= len(self.furnace_actor.x):
            self.dataset_idx = 0

        return obs
    
if __name__ == '__main__':

    split_csv_list = glob("../prediccion-horno/generados/inicial/*.csv")
    split_csv_list = model.sorted_alphanumeric(split_csv_list)

    train, test = np.split(split_csv_list, [int(len(split_csv_list)*0.80)])
    x_train, y_train = model.FeatureEngineering(train)

    x_train = x_train.to_numpy(dtype=np.float32)
    y_train = y_train.to_numpy(dtype=np.float32)

    register(
        id='furnace-agent-v0',                               
        entry_point='agent_model_env:FurnaceEnv',
        kwargs={'train_x': x_train, 'train_y': y_train, 'row_per_episode': 1}
    )

    env = gym.make('furnace-agent-v0')

    obs, info = env.reset()
    print(obs)

    rand_action = env.action_space.sample()
    print(rand_action)
    obs, reward, terminated, _, _ = env.step(rand_action)

    if(terminated):
        obs, info = env.reset()