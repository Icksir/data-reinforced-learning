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

class Action(Enum):
    INCREASE=0
    DECREACE=1
    MANTAIN=2

class FurnaceModel:

    def __init__(self, train_x, train_y):
        self.x = train_x
        self.y = train_y
        self.reset()

        self.last_action=''

    def reset(self, potencia_inicial = 100, seed = None):
        # Inicializa la potencia
        self.potencia = potencia_inicial

    def perform_action(self, action:Action):
        self.last_action = action

        if action == Action.INCREASE:
            self.potencia += 1
        elif action == Action.DECREACE:
            if self.potencia >= 0:
                self.potencia -= 1
        elif action == Action.MANTAIN:
            self.potencia = self.potencia   

def FeatureEngineering(dataset):
    
    # Solo considera el primer split
    df = pd.read_csv(dataset[0])
    df = df.drop(columns=['date_time', 'elapsed_time_seconds', 'elapsed_time', 'consumo', 'cumulative_elapsed_time'])
    y = df.pop('energia_tot')

    # Contatena todos los splits
    '''
    for i in dataset[1:]:
        aux_df = pd.read_csv(i)
        aux_df = aux_df.drop(columns=['date_time', 'elapsed_time_seconds', 'elapsed_time', 'consumo', 'cumulative_elapsed_time'])
        aux_y = aux_df.pop('energia_tot')

        df = pd.concat([df, aux_df], ignore_index=True)
        y = pd.concat([y, aux_y], ignore_index=True)
    '''
        
    return df, y

def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)


if __name__ == '__main__':

    split_csv_list = glob("../prediccion-horno/generados/inicial/*.csv")
    split_csv_list = sorted_alphanumeric(split_csv_list)

    train, test = np.split(split_csv_list, [int(len(split_csv_list)*0.80)])
    x_train, y_train = FeatureEngineering(train)

    x_train = x_train.to_numpy()
    y_train = y_train.to_numpy()

    print(x_train.shape, y_train.shape)

    '''
    furnaceAgent = FurnaceModel(x_train, y_train)

    rand_action = random.choice(list(Action))
    print(rand_action)
    furnaceAgent.perform_action(rand_action)
    print(furnaceAgent.potencia)
    '''
