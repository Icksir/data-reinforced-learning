import numpy as np
import gymnasium as gym
from gymnasium import spaces
import random
import numpy as np
from glob import glob
from gymnasium.envs.registration import register
import agent_model as model

from time import sleep

def check_group(last_action, accion_esperada):
    increase = [model.Action.INCREASE_2, model.Action.INCREASE_5, model.Action.INCREASE_10]
    decrease = [model.Action.DECREACE_2, model.Action.DECREACE_5, model.Action.DECREACE_10]

    if last_action in increase and accion_esperada in increase:
        return 0.2
    elif last_action in decrease and accion_esperada in decrease:
        return 0.2
    
    return 0

class FurnaceEnv(gym.Env):

    metadata = {"render_modes": [None], 'render_fps': 1}

    def __init__(self, datasets):
        super().__init__()

        # Inicializar el actor
        self.furnace_actor = model.FurnaceModel(datasets)

        # Crear action space con las acciones
        self.action_space = gym.spaces.Discrete(len(model.Action))

        # Crear observation space
        self.observation_space = spaces.Box(0, 1e10, shape=(7,), dtype=np.float32)

        # Initialize parameters
        self.dataset_idx = 0

    def reset(self, seed=None, options={}):
        super().reset(seed=seed, options=options)

        self.furnace_actor.reset(seed=seed)
        info = {"potencia": self.furnace_actor.potencia}
        obs = self._next_obs()

        return obs, info

    def step(self, action):

        # Realiza acción
        self.furnace_actor.perform_action(model.Action(action))

        # Recompensa en función de la acción tomada
        reward=0
        if self.furnace_actor.last_action == self.accion_esperada:
            reward += 0.7
        else:
            reward += check_group(self.furnace_actor.last_action, self.accion_esperada)

        # Obtiene observación siguiente
        obs = self._next_obs()

        # Revisa si se terminó la acción y da recompensa por ello
        terminated=False
        if self.terminated:
            if self.consumo_max < 370:
                reward += 5
            else:
                reward += -10
            terminated = True
        
        # Devuelve observación, reward, terminated, truncated, info
        return obs, reward, terminated, False, {"potencia": self.furnace_actor.potencia}
    
    # Retorna la acción esperada en base a los valores de potencia (valor heurístico)
    def get_accion_esperada(self, potencia_inicial, potencia_final):
        accion_esperada = 0
        
        if potencia_final == 0:
            accion_esperada = model.Action.SET_ZERO
        elif (potencia_final - potencia_inicial) == 0:
            accion_esperada = model.Action.MANTAIN
        elif (potencia_final - potencia_final) > 80:
            accion_esperada = model.Action.SET_100
        elif 5 < (potencia_final - potencia_inicial):
            accion_esperada = model.Action.INCREASE_10
        elif 2 < (potencia_final - potencia_inicial) <= 5:
            accion_esperada = model.Action.INCREASE_5
        elif 1 <= (potencia_final - potencia_inicial) <= 2:
            accion_esperada = model.Action.INCREASE_2
        elif -1 >= (potencia_final - potencia_inicial) >= -2:
            accion_esperada = model.Action.DECREACE_2
        elif -2 > (potencia_final - potencia_inicial) >= -5:
            accion_esperada = model.Action.DECREACE_5
        elif -5 > (potencia_final - potencia_inicial):
            accion_esperada = model.Action.DECREACE_10

        return accion_esperada

    def _next_obs(self):

        obs = self.furnace_actor.df.iloc[self.dataset_idx]
        potencia_esperada = self.furnace_actor.df.iloc[self.dataset_idx]['potencia']

        if obs['finished'] != 1:
            potencia_futura = self.furnace_actor.df.iloc[self.dataset_idx + 1]['potencia']
        else:
            potencia_futura = 0

        self.accion_esperada = self.get_accion_esperada(potencia_esperada, potencia_futura)

        self.dataset_idx += 1
        if self.dataset_idx >= len(self.furnace_actor.df):
            self.dataset_idx = 0

        self.terminated = obs['finished']
        self.consumo_max = obs['consumo_max']

        obs = obs.drop(labels=['potencia', 'finished', 'consumo_max'])
        obs = obs.to_numpy(dtype=np.float32)

        return obs
    
if __name__ == '__main__':

    split_csv_list = glob("../prediccion-horno/generados/inicial/*.csv")
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
        print(model.Action(rand_action))
        obs, reward, terminated, _, info = env.step(rand_action)

        print(obs, reward, terminated, info)

        if(terminated):
            obs, info = env.reset()
        
        sleep(1)