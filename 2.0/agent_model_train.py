import gymnasium as gym
import numpy as np
import argparse
import stable_baselines3
import os
import agent_model as model
from glob import glob
from gymnasium.envs.registration import register
import pandas as pd
from time import sleep

#  Train using StableBaseline3
def train(env, sb3_algo):
    model_dir = "models"
    log_dir = "logs"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Start another cmd prompt and launch Tensorboard: tensorboard --logdir logs
    model = sb3_class('MlpPolicy', env, verbose=1, device='cuda', tensorboard_log=log_dir)

    TIMESTEPS = 10000
    iters = 0
    while True:
        iters += 1

        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False)
        model.save(f"{model_dir}/{sb3_algo}_{TIMESTEPS*iters}")

# Test using StableBaseline3. Lots of hardcoding for simplicity.
def test(env, path_to_model):        
    model = sb3_class.load(path_to_model, env=env)

    potencias = []

    obs, info = env.reset() 
    potencias.append(info["potencia"])
    while True:
        action, _ = model.predict(obs)
        obs, _, terminated, _, info = env.step(action)

        print(f'Acción: {action}')
        print(info)

        potencias.append(info)

        if terminated:
            break

        sleep(2)


if __name__ == '__main__':

    split_csv_list = glob("../../prediccion-horno/generados/inicial/*.csv")
    split_csv_list = model.sorted_alphanumeric(split_csv_list)

    register(
        id='furnace-agent-v0',                               
        entry_point='agent_model_env:FurnaceEnv',
        kwargs={'datasets': split_csv_list}
    )

    parser = argparse.ArgumentParser(description='Train or test model.')
    parser.add_argument('sb3_algo', help='StableBaseline3 RL algorithm i.e. A2C, DDPG, DQN, PPO, SAC, TD3')
    parser.add_argument('-t', '--train', action='store_true')
    parser.add_argument('-s', '--test', metavar='path_to_model')
    args = parser.parse_args()

    sb3_class = getattr(stable_baselines3, args.sb3_algo)

    if args.train:
        env = gym.make('furnace-agent-v0')
        train(env, args.sb3_algo)

    if(args.test):
        if os.path.isfile(args.test):
            env = gym.make('furnace-agent-v0')
            test(env, path_to_model=args.test)
        else:
            print(f'{args.test} not found.')
