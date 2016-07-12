import numpy as np
from random import randint
import pylab as pl
import chainer
from chainer import cuda, Function, gradient_check, Variable, optimizers, serializers, utils
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
import minist_data as test_case
import gym
from gym.spaces import Discrete, Box
import random

class NeuralNetwork(Chain):
    def __init__(self, in_size, out_size):
        hidden_size = int((in_size + out_size)/2 + 1)
        super(NeuralNetwork, self).__init__(
            l1=L.Linear(in_size, hidden_size),
            l2=L.Linear(hidden_size, hidden_size),
            l3=L.Linear(hidden_size, out_size)
        )

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        y = self.l3(h2)
        return y

epsilon = .2
learning_rate = .2
env = gym.make('Pendulum-v0')

input_count = env.observation_space.n + env.action_space.n
discount_factor = .99
Q = NeuralNetwork(input_count ,1)

observation = env.reset()
while True:
    action = None
    if epsilon < random.random():
        action_rewards = [Q(observation + a) for a in range(env.action_space)]
        action = np.argmax(action_rewards)
    else:
        action = env.action_space.sample()

    q_t = Q(observation + action)
    (observation, reward, done, _info) = env.step(action)


    Q_update = q_t + learning_rate * (reward + discount_factor * a )


    if done:
        env.reset()








