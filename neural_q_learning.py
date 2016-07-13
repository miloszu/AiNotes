from pybrain.supervised.trainers import BackpropTrainer
from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.tools.customxml import NetworkWriter
from pybrain.tools.customxml import NetworkReader
import pickle

import numpy as np
from random import randint
import pylab as pl
import gym
import random

save_file = True
from_file = False
render = False
network_filename = "CartPole-v0.xml"
epsilon = .6
learning_rate = .5
episode_steps = 200
env = gym.make('CartPole-v0')

observation_count = env.observation_space.shape[0]
action_count = env.action_space.n - 1 #env.action_space.shape[0]
input_count = observation_count + action_count
discount_factor = .99

net = buildNetwork(input_count, input_count, 1, bias=True)

if from_file:
    fileObject = open(network_filename, 'r')
    net = NetworkReader.readFrom(network_filename)

observation = env.reset()
counter =0
while True:
    counter+=1
    env.reset()
    total_reward = 0
    ds = SupervisedDataSet(input_count, 1)
    #print("Q: ", end="")
    for step in range(episode_steps):
        action = None
        if epsilon < random.random():
            actions = [np.append(observation, a) for a in range(0, action_count + 1)]
            print("Q actions: ", actions)
            action_rewards = [net.activate(act)[0] for act in actions]
            action = np.argmax(action_rewards)
        else:
            action = env.action_space.sample()

        x = np.append(observation, action)
        q_t = net.activate(x)
        (observation, reward, done, _info) = env.step(action)
        total_reward += reward
        if step %4 == 0 and render:
            env.render()

        actions = [np.append(observation, a) for a in range(0, action_count + 1)]
        rewards_t = [net.activate(act)[0] for act in actions]
        q_t_max_a = max(rewards_t)
        #print(q_t, end="")
        y = q_t + learning_rate * (reward + discount_factor * q_t_max_a - q_t )
        #print("Reward: {0} Q = {1} <= {2}".format(reward,q_t, y))
        ds.addSample(x, y)
        # if done:
        #     break

    trainer = BackpropTrainer(net, ds)
    print("Total reward: ", total_reward, "Error: ", trainer.train(), "; Counter: ", counter)

    if save_file and counter % 50 == 0:
        NetworkWriter.writeToFile(net, network_filename)
        print("Neural Network Saved to ", network_filename)











