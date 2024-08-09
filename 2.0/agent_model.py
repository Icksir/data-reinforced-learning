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

def FeatureEngineering(dataset):
    
    df = pd.read_csv(dataset[0])
    df = df.drop(columns=['date_time', 'elapsed_time_seconds', 'elapsed_time', 'cumulative_elapsed_time'])
    df['finished'] = 0
    df.loc[df.index[-1], 'finished'] = 1
    df['finished'] = df['finished'].astype(int)
    df['consumo_max'] = 0.0
    df.loc[df.index[-1], 'consumo_max'] = df['consumo'].max()
    df['potencia'] = df['potencia'].astype(int)
    df = df.drop(columns=['consumo'])
    # df = df.drop(columns=['potencia', 'finished', 'consumo_max'])

    df = pd.DataFrame(data={'tiempo': df['tiempo'],
                            'carga': df['carga_ton'],
                            'oxigeno': df['oxigeno'],
                            'carbon_inicial': df['carbon'].loc[0],
                            'carbon': df['carbon'],
                            'potencia': df['potencia'],
                            'finished': df['finished'],
                            'consumo_max': df['consumo_max']})
    
    df.loc[df.index[0], 'carbon_inicial'] = df['carbon'].loc[0]

    for i in dataset[1:]:
        aux_df = pd.read_csv(i)
        aux_df = aux_df.drop(columns=['date_time', 'elapsed_time_seconds', 'elapsed_time', 'cumulative_elapsed_time'])
        aux_df['finished'] = 0
        aux_df.loc[aux_df.index[-1], 'finished'] = 1
        aux_df['finished'] = aux_df['finished'].astype(int)
        aux_df['consumo_max'] = 0.0
        aux_df.loc[aux_df.index[-1], 'consumo_max'] = aux_df['consumo'].max()
        aux_df['potencia'] = aux_df['potencia'].astype(int)
        aux_df = aux_df.drop(columns=['consumo'])

        aux_df = pd.DataFrame(data={'tiempo': aux_df['tiempo'],
                                    'carga': aux_df['carga_ton'],
                                    'oxigeno': aux_df['oxigeno'],
                                    'carbon_inicial': aux_df['carbon'].loc[0],
                                    'carbon': aux_df['carbon'],
                                    'potencia': aux_df['potencia'],
                                    'finished': aux_df['finished'],
                                    'consumo_max': aux_df['consumo_max']})

        df = pd.concat([df, aux_df], ignore_index=True)
    
    return df

def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)

class FurnaceModel:

    def __init__(self, datasets):
        self.df = FeatureEngineering(datasets)
        self.reset(0, 0, 0, 0, 0, 0)
        self.last_action = None

    def reset(self, tiempo, carga, oxigeno, carbon_inicial, carbon, potencia, seed = None):
        
        # Resetea todas las variables para que sean las mismas que la primera fila del split

        self.tiempo = tiempo
        self.carga = carga
        self.oxigeno = oxigeno
        self.carbon_inicial = carbon_inicial 
        self.carbon = carbon
        self.potencia = potencia

    def perform_action(self, action):

        # IMPORTANTE: Estos son valores relativos, por eso se suman
        # Se considera esto para

        # oxigeno va entre 0 y 140
        if 0 <= action <= 140:
            self.oxigeno += action
            self.last_action = 'oxigeno'

        # carbon va entre 141 y 188
        elif 141 <= action <= 188:

            valor = action - 141

            if 0 <= valor <= 30:
                self.carbon += valor
            else:
                mult = valor - 30
                self.carbon += mult * 50

            self.last_action = 'carbon'

        elif 189 <= action <= 418:
            valor = action - 189
            self.potencia += valor

            self.last_action = 'potencia'

if __name__ == '__main__':

    split_csv_list = glob("../../prediccion-horno/generados/inicial/*.csv")
    split_csv_list = sorted_alphanumeric(split_csv_list)

    agente = FurnaceModel(split_csv_list)

    # print(agente.df.shape)
    # print(agente.df)
    while True:
        rand_action = random.randint(0, 418)
        print(f"Action: {rand_action}")
        print(f"Oxigeno: {agente.oxigeno} | Carbon: {agente.carbon} | Potencia: {agente.potencia}")
        agente.perform_action(rand_action)
        print(f"Oxigeno: {agente.oxigeno} | Carbon: {agente.carbon} | Potencia: {agente.potencia}")

        time.sleep(1)
