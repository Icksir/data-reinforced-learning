import numpy as np
import gymnasium as gym
from gymnasium import spaces
import random
import numpy as np
from glob import glob
from gymnasium.envs.registration import register
import agent_model as model

from time import sleep
import math

def casos_carbon():

    casos_carbon_inicial={}

    casos_carbon_inicial[400] = 0
    casos_carbon_inicial[800] = 1
    casos_carbon_inicial[600] = 2
    casos_carbon_inicial[1000] = 3
    casos_carbon_inicial[2000] = 4
    casos_carbon_inicial[692] = 5
    casos_carbon_inicial[0] = 6
    casos_carbon_inicial[1700] = 7
    casos_carbon_inicial[515] = 8
    casos_carbon_inicial[533] = 9
    casos_carbon_inicial[306] = 10
    casos_carbon_inicial[1400] = 11
    casos_carbon_inicial[700] = 12
    casos_carbon_inicial[720] = 13
    casos_carbon_inicial[500] = 14
    casos_carbon_inicial[200] = 15
    casos_carbon_inicial[446] = 16
    casos_carbon_inicial[1200] = 17

    return casos_carbon_inicial

class FurnaceEnv(gym.Env):

    metadata = {"render_modes": [None], 'render_fps': 1}

    def __init__(self, datasets):
        super().__init__()

        # Inicializar el actor
        self.furnace_actor = model.FurnaceModel(datasets)

        # Crear action space con las acciones
        '''
        Son valores relativos

        oxigeno -> 141 valores
        carbon -> 47 valores
        potencia -> 231 valores
        '''
        self.action_space = gym.spaces.Discrete(419)

        # Crear observation space con
        '''
        Para las observaciones son valores absolutos

        tiempo
        carga
        oxigeno
        carbon_inicial
        carbon actual
        potencia real
        '''
        self.observation_space = spaces.Box(low=np.array([0, 50,0,0,0,0]), high=np.array([565, 230, 7582, 2100, 3743, 124]), dtype=np.float32)

        # Initialize parameters
        self.dataset_idx = 0

        # Tiempos anteriores
        self.tiempo_anterior = None
        self.carga_anterior = None
        self.oxigeno_anterior = None
        self.carbon_anterior = None
        self.potencia_anterior = None

    def reset(self, seed=None, options={}):
        super().reset(seed=seed, options=options)
        obs = self._next_obs()

        self.furnace_actor.reset(0, 120, 0, 0, 0, 0, seed=seed)
        info = {"potencia": self.furnace_actor.potencia}

        self.tiempo_anterior = None
        self.carga_anterior = None
        self.oxigeno_anterior = None
        self.carbon_anterior = None
        self.potencia_anterior = None

        return obs, info

    def step(self, action):

        # Realiza acción
        self.furnace_actor.perform_action(action)

        # Obtiene observación siguiente
        obs = self._next_obs()

        reward = 0

        # Recompensa por acción idéntica

        if self.furnace_actor.last_action == 'oxigeno':

            if self.delta_oxigeno == action:
                reward += 5
        
        elif self.furnace_actor.last_action == 'carbon':
            
            valor = action - 141

            if valor>=30:
                valor = int(math.floor((valor-30))/50+30)

            if self.delta_carbon == valor:
                reward += 5

        elif self.furnace_actor.last_action == 'potencia':

            valor = action - 189

            if self.delta_potencia == valor:
                reward += 5

        # Revisa si se terminó la acción y da recompensa por ello
        terminated=False
        if self.terminated:

            if self.consumo_max < 300:
                reward += 200
            elif self.consumo_max < 320:
                reward += 150
            elif self.consumo_max < 350:
                reward += 90
            elif self.consumo_max < 390:
                reward += 40
            elif self.consumo_max < 410:
                reward += 10
            else:
                reward += 1

            terminated = True
        
        # Devuelve observación, reward, terminated, truncated, info
        return obs, reward, terminated, False, {"oxigeno": self.furnace_actor.oxigeno, 
                                                "carbon": self.furnace_actor.carbon,
                                                "potencia": self.furnace_actor.potencia}

    def _next_obs(self):

        obs = self.furnace_actor.df.iloc[self.dataset_idx]

        if self.tiempo_anterior == None:
            self.tiempo_anterior = 0
            self.carga_anterior = 0
            self.oxigeno_anterior = 0
            self.carbon_anterior = 0
            self.potencia_anterior = 0

        self.delta_carga = int(obs['carga'] - self.carga_anterior)
        self.delta_oxigeno = int(obs['oxigeno'] - self.oxigeno_anterior)
        self.delta_carbon = int(obs['carbon'] - self.carbon_anterior)
        self.delta_potencia = int(obs['potencia'] - self.potencia_anterior)

        if self.delta_carbon>=30:
            self.delta_carbon = int(math.floor((self.delta_carbon-30))/50+30)


            #4: carbon ->  [0,29] se mapea al mismo numero
            #              [30,79] se mapea a 30
            #              [80,129] se mapea a  31
            # etc
        
        # El uso de los deltas (acciones) como observaciones no debería ser utilizado en mi opinión
        self.delta_oxigeno_normalizado = self.delta_oxigeno/140
        self.delta_carbon_normalizado = self.delta_carbon/46
        self.delta_potencia_normalizada = abs(self.delta_potencia)/115

        # print(f"    Delta oxigeno: {self.delta_oxigeno} | Delta carbon: {self.delta_carbon} | Delta potencia: {self.delta_potencia}")

        self.dataset_idx += 1
        if self.dataset_idx >= len(self.furnace_actor.df):
            self.dataset_idx = 0

        self.terminated = obs['finished']
        self.consumo_max = obs['consumo_max']

        self.tiempo_anterior = obs['tiempo']
        self.carga_anterior = obs['carga']
        self.oxigeno_anterior = obs['oxigeno']
        self.carbon_anterior = obs['carbon']
        self.potencia_anterior = obs['potencia']

        obs = obs.drop(labels=['finished', 'consumo_max'])
        obs = obs.to_numpy(dtype=np.float32)

        # Las acciones no se pueden codificar, la gracia es que el modelo aprenda en base a las recompensas

        return obs
    
if __name__ == '__main__':

    split_csv_list = glob("../../prediccion-horno/generados/inicial/*.csv")
    split_csv_list = model.sorted_alphanumeric(split_csv_list)

    register(
        id='furnace-agent-v0',                               
        entry_point='agent_model_env:FurnaceEnv',
        kwargs={'datasets': split_csv_list}
    )

    '''
    En dataset:
    0: tiempo
    1: energia_tot
    2: oxigeno
    3: lncoix
    4: injx_carb
    5: carbon
    6: carga_ton
    '''

    env = gym.make('furnace-agent-v0')

    obs, info = env.reset()

    while True:
        rand_action = env.action_space.sample()
        print("Acción random ------------------")
        print(f"    Accion escogida: {rand_action}")
        obs, reward, terminated, _, info = env.step(rand_action)
        print('\n')

        # print(obs, reward, terminated, info)

        if(terminated):
            obs, info = env.reset()
        
        sleep(2)