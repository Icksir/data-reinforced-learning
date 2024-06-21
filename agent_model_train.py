import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import random
import pickle
from stable_baselines3 import A2C
import os
import agent_model_env
import agent_model as model
from glob import glob
from gymnasium.envs.registration import register

#  Train using StableBaseline3
def train_sb3():
    # Where to store trained model and logs
    model_dir = "models"
    log_dir = "logs"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    env = gym.make('furnace-agent-v0')

    # Use Advantage Actor Critic (A2C) algorithm.
    # Use MlpPolicy for observation space 1D vector.
    model = A2C('MlpPolicy', env, verbose=1, device='cuda', tensorboard_log=log_dir)
   
    # This loop will keep training until you stop it with Ctr-C.
    # Start another cmd prompt and launch Tensorboard: tensorboard --logdir logs
    # Once Tensorboard is loaded, it will print a URL. Follow the URL to see the status of the training.
    # Stop the training when you're satisfied with the status.
    TIMESTEPS = 1000
    iters = 0
    while True:
        iters += 1

        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False) # train
        model.save(f"{model_dir}/a2c_{TIMESTEPS*iters}") # Save a trained model every TIMESTEPS

# Test using StableBaseline3. Lots of hardcoding for simplicity.
def test_sb3(render=True):

    env = gym.make('warehouse-robot-v0', render_mode='human' if render else None)

    # Load model
    model = A2C.load('models/a2c_2000', env=env)

    # Run a test
    obs = env.reset()[0]
    terminated = False
    while True:
        action, _ = model.predict(observation=obs, deterministic=True) # Turn on deterministic, so predict always returns the same behavior
        obs, _, terminated, _, _ = env.step(action)

        if terminated:
            break

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

    # Train/test using StableBaseline3
    train_sb3()
    # test_sb3()
