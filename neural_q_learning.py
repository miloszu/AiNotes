from pybrain.supervised.trainers import BackpropTrainer
from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.tools.customxml import NetworkWriter
from pybrain.tools.customxml import NetworkReader
from gym.spaces import Discrete, Box


import numpy as np
import gym
import random

save_file = True
from_file = False
render = False
network_filename = "Acrobot-v0.4layers.xml"
epsilon = 0.9
learning_rate = .9
episode_steps = 1000
avg_steps_count = 25
env = gym.make('Acrobot-v0')



observation_count = env.observation_space.shape[0]
action_count =  None
if isinstance(env.action_space, Discrete):
    action_count= 1
else:
    action_count=env.action_space.shape[0]

input_count = observation_count + action_count
discount_factor = .99

net = buildNetwork(input_count, input_count, input_count, 1, bias=True)

if from_file:
    fileObject = open(network_filename, 'r')
    net = NetworkReader.readFrom(network_filename)

observation = env.reset()
counter =0
max_reward = -9999999
min_reward = 9999999
avg_queue = []
while True:
    counter+=1
    env.reset()
    total_reward = 0
    ds = SupervisedDataSet(input_count, 1)
    #print("Q: ", end="")
    for step in range(episode_steps):
        action = None
        if epsilon < random.random():
            actions = [np.append(observation, a) for a in range(0, env.action_space.n)]
            action_rewards = [net.activate(act)[0] for act in actions]
            #print("Q rewards: ", action_rewards)
            action = np.argmax(action_rewards)
        else:
            action = env.action_space.sample()

        x = np.append(observation, action)
        q_t = net.activate(x)
        (observation, reward, done, _info) = env.step(action)
        total_reward += reward

        if step %4 == 0 and render:
            env.render()
        actions = [np.append(observation, a) for a in range(0, env.action_space.n)]
        rewards_t = [net.activate(act)[0] for act in actions]
        q_t_max_a = max(rewards_t)
        #print(q_t, end="")
        y = q_t + learning_rate * (reward + discount_factor * q_t_max_a - q_t )
        #print("Reward: {0} Q = {1} <= {2}".format(reward,q_t, y))
        ds.addSample(x, y)
        if done:
            break
    avg_queue.append(total_reward)
    if len(avg_queue) > avg_steps_count:
        avg_queue.pop(0)
    if max_reward < total_reward:
        max_reward = total_reward
    if min_reward > total_reward:
        min_reward = total_reward
    if max_reward == min_reward:
        epsilon = 0.9
    else:
        walking_avg = sum(avg_queue)/len(avg_queue)
        epsilon = max((max_reward - walking_avg) / (max_reward - min_reward) * 0.9, 0.01)

    trainer = BackpropTrainer(net, ds)
    print("Total reward: ", total_reward, "Error: ", trainer.train(), "Epsilon: ", epsilon, "Counter: ", counter)

    if save_file and counter % 50 == 0:
        NetworkWriter.writeToFile(net, network_filename)
        print("Neural Network Saved to ", network_filename)
