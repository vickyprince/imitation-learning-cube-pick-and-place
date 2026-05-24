import gymnasium as gym
import gym_xarm
import time
import numpy as np

env = gym.make(
    "gym_xarm/XarmLift-v0",
    render_mode="human"
)

obs, info = env.reset()

for _ in range(500):

    action = np.array([0.2, 0.0, 0.0, 0.0])

    obs, reward, terminated, truncated, info = env.step(action)

    time.sleep(0.02)

env.close()