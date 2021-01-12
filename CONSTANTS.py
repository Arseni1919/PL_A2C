import numpy as np
import os
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import random_split
from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from collections import namedtuple, deque
import gym

ENV = "CartPole-v0"  # gym environment tag
# ENV = 'LunarLander-v2'
# ENV='MountainCar-v0'
NUMBER_OF_GAMES = 5
SAVE_WEIGHTS = True
# ------------------------------------------- #
# ------------------FOR ALG:----------------- #
# ------------------------------------------- #

MAX_EPOCHS = 300  # maximum epoch to execute
BATCH_SIZE = 64  # size of the batches
LR = 1e-3  # learning rate
GAMMA = 0.99  # discount factor
SYNC_RATE = 10  # how many frames do we update the target network
REPLAY_SIZE = 1000  # capacity of the replay buffer
WARM_START_STEPS = REPLAY_SIZE  # how many samples do we use to fill our buffer at the start of training

# EPISODE_LENGTH = 200  # max length of an episode
# MAX_EPISODE_REWARD = 200  # max episode reward in the environment

EPS_LAST_FRAME = int(REPLAY_SIZE / BATCH_SIZE * MAX_EPOCHS)  # what frame should epsilon stop decaying
EPS_START = 1  # starting value of epsilon
EPS_END = 0.01  # final value of epsilon

Experience = namedtuple('Experience', field_names=['state', 'action', 'reward', 'done', 'new_state'])
