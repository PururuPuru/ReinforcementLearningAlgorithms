import numpy as np
import random
import sys
import gym

from collections import defaultdict
import matplotlib.pyplot as plt

import check_test
from plot_utils import plot_values

cliff_env = gym.make('CliffWalking-v0')


def get_e_policy(env, svalue, epsilon):
    if random.random() > epsilon:  # select greedy action with probability epsilon
        return np.argmax(svalue)
    else:  # otherwise, select an action randomly
        return random.choice(np.arange(env.action_space.n))


def sarsa(env, num_episodes, alpha=0.01, gamma=1.0):
    value = defaultdict(lambda: np.zeros(env.action_space.n))
    for i_episode in range(1, num_episodes + 1):
        if i_episode % 100 == 0:
            print("\rEpisode {}/{}".format(i_episode, num_episodes), end="")
            sys.stdout.flush()
        epsilon = (1.0 / i_episode)
        state = env.reset()
        action = get_e_policy(env, value[state], epsilon)
        while True:
            next_state, reward, done, info = env.step(action)
            next_action = get_e_policy(env, value[next_state], epsilon)
            value[state][action] += alpha * (reward + gamma * value[next_state][next_action] - value[state][action])
            state = next_state
            action = next_action
            if done:
                break

    return value


# obtain the estimated optimal policy and corresponding action-value function
Q_sarsa = sarsa(cliff_env, 5000)

# print the estimated optimal policy
policy_sarsa = np.array([np.argmax(Q_sarsa[key]) if key in Q_sarsa else -1 for key in np.arange(48)]).reshape(4, 12)
check_test.run_check('td_control_check', policy_sarsa)
print("\nEstimated Optimal Policy (UP = 0, RIGHT = 1, DOWN = 2, LEFT = 3, N/A = -1):")
print(policy_sarsa)

# plot the estimated optimal state-value function
V_sarsa = ([np.max(Q_sarsa[key]) if key in Q_sarsa else 0 for key in np.arange(48)])
plot_values(V_sarsa)
