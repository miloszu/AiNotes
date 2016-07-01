import numpy as np
from gym import Env


class HillClimbing:
    def __init__(self, environment: Env, episode_count=2000, episode_steps=2000, render_training=False):
        self._episode_steps = episode_steps
        self._episode_count = episode_count
        self._environment = environment
        self._render_training = render_training

    @staticmethod
    def _compute_action(observation, policy):
        return np.dot(np.append(observation, np.array([1])), policy) > 0

    def _random_policy(self):
        return np.random.uniform(-1, 1, self._environment.observation_space.shape[0] + 1)

    def _mutate(self, policy, mutation_factor):
        if policy is None:
            return self._random_policy()
        return policy + \
               (np.random.uniform(-1, 1, size=self._environment.observation_space.shape[0] + 1) * mutation_factor)

    def _run_episode(self, policy, render=False):
        episode_reward = 0
        observation = self._environment.reset()
        for t in range(0, self._episode_steps):
            if render or self._render_training:
                self._environment.render()
            action = self._compute_action(observation, policy)
            observation, reward, done, info = self._environment.step(action)
            episode_reward += reward
            if done:
                break
        return observation, episode_reward

    def run(self):
        best_policy = None
        max_reward = -self._episode_steps
        min_reward = self._episode_steps
        for i_episode in range(0, self._episode_count):
            mutation_factor = (((2 * self._episode_steps) - np.fabs(max_reward - min_reward)) /
                               (2 * self._episode_steps)) * .5
            tested_policy = self._mutate(best_policy, mutation_factor)
            observation, episode_reward = self._run_episode(tested_policy)
            if max_reward < episode_reward:
                best_policy = tested_policy
                max_reward = episode_reward
                print("New max reward: {0}".format(max_reward))
            if min_reward >= episode_reward:
                min_reward = episode_reward
                best_policy = self._random_policy()
                print("Min reward - new random policy.")
            if max_reward >= self._episode_steps:
                print("Max reward reached.")
                break
            print("Episode reward: {0}. Max Reward {1}. Mutation factor: {2}"
                  .format(episode_reward, max_reward, mutation_factor))
        print("Best policy:")
        print(best_policy)
        obs, reward = self._run_episode(best_policy, True)
        print("Best policy reward: {0}. ".format(reward))
