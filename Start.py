import hill_climbing
import gym

cart_pole_env = gym.make('Acrobot-v0')
cart_pole_hill = hill_climbing.HillClimbing(cart_pole_env, episode_count=2000, episode_steps=1000)
cart_pole_hill.run()


# acrobot_env = gym.make('Acrobot-v0')
# acrobot_env_hill = hill_climbing.HillClimbing(acrobot_env, episode_steps= 500, render_training=False)
# acrobot_env_hill.run()