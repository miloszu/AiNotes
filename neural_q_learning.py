import neural_network
from gym.spaces import Discrete, Box
import numpy as np
import gym
import random

save_file = True
from_file = False
render = False
train = True
network_filename = "MountainCar-v0"
epsilon = 1
episode_steps = 1000
alpha = .2
env = gym.make('MountainCar-v0')


observation_count = env.observation_space.shape[0]
action_count =  None
if isinstance(env.action_space, Discrete):
    action_count= env.action_space.n
else:
    action_count=env.action_space.shape[0]

discount_factor = .99

net = neural_network.NeuralNetwork([neural_network.TanhLayer(observation_count),
                                    neural_network.TanhLayer(observation_count),
                                    neural_network.TanhLayer(action_count)])

# if from_file:
#     fileObject = open(network_filename, 'r')
#     net = NetworkReader.readFrom(network_filename)

counter =0
minibatch_count = 1100
max_reward=None
replay_memory=[]
replay_memory_size = 10000;
while True:
    counter+=1
    observation = env.reset()
    total_reward = 0
    total_error = 0
    print(net.predict(observation))
    for step in range(episode_steps):
        q_values = net.predict(observation)
        if epsilon < random.random():
            max_q = max(q_values)
            action = np.argmax(q_values)
        else:
            action = env.action_space.sample()
            max_q = q_values[action]

        (observation_prim, reward, done, _info) = env.step(action)
        reward *= .0001
        total_reward += reward
        if step %4 == 0 and render:
            env.render()

        if step == episode_steps - 1:
            done = True
        replay_memory.append((observation, action, reward, observation_prim, done))
        if replay_memory_size > len(replay_memory):
            replay_memory.pop(0)
        observation = observation_prim
        if done:
            break

    if max_reward is None or max_reward < total_reward:
        max_reward = total_reward

    if len(replay_memory) > minibatch_count:
        random_minibatch = random.sample(replay_memory, minibatch_count)
        X = np.empty((0, observation_count))
        Y = np.empty((0, action_count))

        for (o, a, r, op, d) in random_minibatch:
            q_v = net.predict(o)
            m_q = q_v[a]
            q_v_prim = net.predict(op)
            m_q_prim = max(q_v_prim)
            if d:
                y = m_q + alpha * (r - m_q)
            else:
                y = m_q + alpha * (r + discount_factor * m_q_prim - m_q)
            q_v[a] = y
            X = np.append(X, np.array([o]), axis=0)
            Y = np.append(Y, np.array([q_v]), axis=0)
        net.fit(X,Y, epochs=minibatch_count*2)
    epsilon -= .001


    print("Steps: ", step, "Total reward: ", total_reward, "Error: ", total_error, "Epsilon: ", epsilon, "Counter: ", counter,
          "Max reward: ", max_reward)

    # if save_file and counter % 50 == 0:
    #     NetworkWriter.writeToFile(net, network_filename)
    #     print("Neural Network Saved to ", network_filename)
