# install: !pip install -q git+https://github.com/deepmind/acme.git#egg=dm-acme[jax,tf,envs]
# works on google colab 

import numpy as np

import gym
from gym.wrappers import RecordVideo

import dm_env
import acme
from acme import wrappers
from acme import specs
from acme.tf import networks
from acme.tf import utils as tf2_utils
from acme.agents.tf import dqn
from acme.utils import loggers

import sonnet as snt

import tensorflow as tf
import tensorflow_probability as tfp

import logging
logging.getLogger().setLevel(logging.DEBUG)  # help with seeing ACME logs

from march_madness import MarchMadnessEnvironment

def train_model():
    environment = MarchMadnessEnvironment(
        filename='data/fivethirtyeight_ncaa_forecasts.csv'
    )

    environment = wrappers.GymWrapper(environment)
    environment = wrappers.SinglePrecisionWrapper(environment)

    network = snt.Sequential([
        snt.Flatten(),
        snt.nets.MLP([50, 50, env_spec.actions.num_values]),
    ])
    env_spec = acme.make_environment_spec(env)

    agent = dqn.DQN(env_spec, network,
                epsilon=0.1,
                learning_rate=1e-3,
                max_replay_size=1_000,
                batch_size=32)

    # logger = loggers.TerminalLogger(time_delta=0.5)
    env_loop = acme.EnvironmentLoop(env, agent)
    env_loop.run(num_episodes=300)

