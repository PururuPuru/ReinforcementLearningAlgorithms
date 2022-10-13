import numpy as np


def policy_evaluation(env, policy, theta=0.00001, gamma=1.0):
    value = np.zeros(env.observation_space.n)
    action_prob = 1 / env.action_space.n
    while True:
        delta = 0
        for s in range(env.observation_space.n):
            next_value = 0
            for a in range(env.action_space.n):
                for prob, next_state, reward, done in env.P[s][policy[s]]:
                    next_value += action_prob * prob * (reward + gamma * value[next_state])
            delta = max(delta, np.abs(value[s] - next_value))
            value[s] = next_value
        if delta < theta:
            break
    return value
