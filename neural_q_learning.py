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

class RectifierNetwork(Chain):
    def __init__(self, in_size, out_size):
        super(RectifierNetwork, self).__init__(
            l1=L.Linear(in_size, 4),
            l2=L.Linear(4, 4),
            l3=L.Linear(4, out_size)
        )

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        y = self.l3(h2)
        return y

epsilon = .2
learning_rate = .2
env = gym.make('CartPole-v0')

observation_count = env.observation_space.shape[0]
action_count = env.action_space.n - 1 #env.action_space.shape[0]
input_count = observation_count + action_count
discount_factor = .99
Q = RectifierNetwork(input_count, 1)
model = L.Classifier(Q)
optimizer = optimizers.SGD()
optimizer.setup(model)

observation = env.reset()
while True:
    action = None
    if epsilon < random.random():
        action_rewards = Q(np.array([np.array(np.append(observation, a)) for a in range(0,action_count + 1)], dtype=np.float32)).data
        action = np.argmax(action_rewards)
    else:
        action = env.action_space.sample()

    x = np.append(observation, action)
    q_t = Q(np.array([x], dtype=np.float32)).data[0][0]
    (observation, reward, done, _info) = env.step(action)
    env.render()

    q_t_max_a = max(Q(np.array([np.array(np.append(observation, a)) for a in range(0,action_count + 1)], dtype=np.float32)).data[0])
    Q_update = q_t + learning_rate * (reward + discount_factor * q_t_max_a - q_t )

    x = Variable(np.array([x], dtype=np.float32))
    y = Variable(np.array([np.array([Q_update])], dtype=np.float32))

    optimizer.update(model, x, y)
    if done:
        env.reset()








