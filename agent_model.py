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
    INCREASE_2=0
    INCREASE_5=1
    INCREASE_10=2
    DECREACE_2=3
    DECREACE_5=4
    DECREACE_10=5
    MANTAIN=6
    SET_ZERO=7
    SET_100=8

class FurnaceModel:

    def __init__(self, datasets):
        self.df = FeatureEngineering(datasets)
        self.reset()

        self.last_action=''
        self.consumo = 0

    def reset(self, potencia_inicial = 100, seed = None):
        # Inicializa la potencia
        self.potencia = potencia_inicial

    def perform_action(self, action:Action):
        self.last_action = action

        if action == Action.INCREASE_2:
            self.potencia += 2
        elif action == Action.INCREASE_5:
            self.potencia += 5
        elif action == Action.INCREASE_10:
            self.potencia += 10
        elif action == Action.DECREACE_2:
            if self.potencia >= 0:
                self.potencia -= 2
        elif action == Action.DECREACE_5:
            if self.potencia >= 0:
                self.potencia -= 5
        elif action == Action.DECREACE_10:
            if self.potencia >= 0:
                self.potencia -= 10
        elif action == Action.MANTAIN:
            self.potencia = self.potencia  
        elif action == Action.SET_ZERO:
            self.potencia = 0 
        elif action == Action.SET_100:
            self.potencia = 100

def FeatureEngineering(dataset):
    
    # Solo considera el primer split
    df = pd.read_csv(dataset[0])
    df = df.drop(columns=['date_time', 'elapsed_time_seconds', 'elapsed_time', 'cumulative_elapsed_time'])

    df['finished'] = 0
    df.loc[df.index[-1], 'finished'] = 1
    df['finished'] = df['finished'].astype(int)

    df['consumo_max'] = 0
    df.loc[df.index[-1], 'consumo_max'] = df['consumo'].max()

    df['potencia'] = df['potencia'].astype(int)

    df = df.drop(columns=['consumo'])

    # Contatena todos los splits
    '''
    for i in dataset[1:]:
        aux_df = pd.read_csv(i)
        aux_df = aux_df.drop(columns=['date_time', 'elapsed_time_seconds', 'elapsed_time', 'consumo', 'cumulative_elapsed_time'])
        aux_y = aux_df.pop('energia_tot')

        df = pd.concat([df, aux_df], ignore_index=True)
        y = pd.concat([y, aux_y], ignore_index=True)
    '''
        
    return df

def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)


if __name__ == '__main__':

    split_csv_list = glob("../prediccion-horno/generados/inicial/*.csv")
    split_csv_list = sorted_alphanumeric(split_csv_list)

    agente = FurnaceModel(split_csv_list)

    # df = df.to_numpy()

    print(agente.df.shape)

    print(agente.df)

    '''
    furnaceAgent = FurnaceModel(x_train, y_train)

    rand_action = random.choice(list(Action))
    print(rand_action)
    furnaceAgent.perform_action(rand_action)
    print(furnaceAgent.potencia)
    '''
