import numpy as np
import random
import gym
import sys

from collections import defaultdict
import matplotlib.pyplot as plt

import check_test
from plot_utils import plot_values

cliff_env = gym.make('CliffWalking-v0')


def get_action_prob(env, svalue, epsilon):
    policy = np.ones(env.action_space.n) * epsilon / env.action_space.n
    policy[np.argmax(svalue)] = 1 - epsilon + (epsilon / env.action_space.n)
    return policy


def get_e_policy(env, svalue, epsilon):
    if random.random() > epsilon:  # select greedy action with probability epsilon
        return np.argmax(svalue)
    else:  # otherwise, select an action randomly
        return random.choice(np.arange(env.action_space.n))


def expected_sarsa(env, num_episodes, alpha=0.01, gamma=1.0):
    value = defaultdict(lambda: np.zeros(env.action_space.n))
    for i_episode in range(1, num_episodes + 1):
        if i_episode % 100 == 0:
            print("\rEpisode {}/{}".format(i_episode, num_episodes), end="")
            sys.stdout.flush()
        epsilon = 0.005
        state = env.reset()

        while True:
            action = get_e_policy(env, value[state], epsilon)
            next_state, reward, done, info = env.step(action)
            weighted_sum = np.dot(value[next_state], get_action_prob(env, value[next_state], epsilon))
            value[state][action] += alpha * (reward + gamma * weighted_sum - value[state][action])
            state = next_state
            if done:
                break

    return value


# obtain the estimated optimal policy and corresponding action-value function
Q_expsarsa = expected_sarsa(cliff_env, 5000, alpha=1)

# print the estimated optimal policy
policy_expsarsa = np.array([np.argmax(Q_expsarsa[key]) if key in Q_expsarsa else -1 for key in np.arange(48)]).reshape(4,12)
check_test.run_check('td_control_check', policy_expsarsa)
print("\nEstimated Optimal Policy (UP = 0, RIGHT = 1, DOWN = 2, LEFT = 3, N/A = -1):")
print(policy_expsarsa)

# plot the estimated optimal state-value function
plot_values([np.max(Q_expsarsa[key]) if key in Q_expsarsa else 0 for key in np.arange(48)])
