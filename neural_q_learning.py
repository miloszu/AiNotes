from pybrain.supervised.trainers import BackpropTrainer
from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.tools.customxml import NetworkWriter
from pybrain.tools.customxml import NetworkReader
from pybrain.structure.modules import ReluLayer, LinearLayer
from gym.spaces import Discrete, Box


import numpy as np
import gym
import random

save_file = False
from_file = False
render = False
train = True
network_filename = "CartPole-v0.xml"
epsilon = .3
episode_steps = 1000
avg_steps_count = 10
alpha = .2
env = gym.make('CartPole-v0')



observation_count = env.observation_space.shape[0]
action_count =  None
if isinstance(env.action_space, Discrete):
    action_count= 1
else:
    action_count=env.action_space.shape[0]

input_count = observation_count + action_count
discount_factor = .99

net = buildNetwork(input_count, input_count, 1, bias=True, hiddenclass=ReluLayer, outclass=LinearLayer)

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
        q_sa = net.activate(x)

        (observation, reward, done, _info) = env.step(action)

        total_reward += reward

        if step %4 == 0 and render:
            env.render()

        actions = [np.append(observation, a) for a in range(0, env.action_space.n)]
        q_values = [net.activate(act)[0] for act in actions]
        max_q_prim = max(q_values)

        #print(q_t, end="")
        if done:
            y = q_sa + alpha * (reward - q_sa)
        else:
            y = q_sa + alpha * (reward + max_q_prim - q_sa)
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
        epsilon = .3
    else:
        walking_avg = sum(avg_queue)/len(avg_queue)
        epsilon = max((max_reward - walking_avg) / (max_reward - min_reward) * .3, 0.00000001)

    if train:
        trainer = BackpropTrainer(net, ds)
        epooch_count = 10
        total_error =0
        for i in range(epooch_count):
            total_error += trainer.train()
        print("Total reward: ", total_reward, "Error: ", total_error/epooch_count, "Epsilon: ", epsilon, "Counter: ",
              counter, "Max reward: ", max_reward, "Min reward: ", min_reward)
    else:
        print("Not trained. Total reward: ", total_reward, "Epsilon: ", epsilon, "Counter: ",
              counter, "Max reward: ", max_reward, "Min reward: ", min_reward)

    if save_file and counter % 50 == 0:
        NetworkWriter.writeToFile(net, network_filename)
        print("Neural Network Saved to ", network_filename)
