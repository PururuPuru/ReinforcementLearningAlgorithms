import numpy as np

from policy_iteration import get_action_values


def value_iteration(env, policy, theta=0.00001):
    value = np.zeros(env.observation_space.n)

    while True:
        delta = 0
        for s in range(env.observation_space.n):
            best_a = get_action_values(env, s, value)
            best_action_value = np.max(best_a)
            delta = max(delta, np.abs(value[s] - best_action_value))
            value[s] = best_action_value
        if delta < theta:
            break

    for s in range(env.observation_space.n):
        best_a = get_action_values(env, s, value)
        best_action = np.argmax(best_a)
        policy[s] = best_action

    return policy, value
