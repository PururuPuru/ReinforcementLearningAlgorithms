import numpy as np

from policy_evaluation import policy_evaluation


def get_action_values(env, state, value, gamma=1.0):
    action_values = np.zeros(env.action_space.n)
    for action in range(env.action_space.n):
        for prob, next_state, reward, done in env.P[state][action]:
            action_values[action] += prob * (reward + gamma * value[next_state])

    return action_values


def get_random_policy(env):
    random_action = np.random.choice(tuple(env.P[0].keys()), len(env.P))
    return {s: a for s, a in enumerate(random_action)}


def policy_iteration(env, policy, gamma=1.0):
    while True:
        value = policy_evaluation(env, policy, gamma=gamma)
        policy_stable = True

        for s in range(env.observation_space.n):
            action_values = get_action_values(env, s, value, gamma)
            best_a = np.argmax(action_values)

            if policy[s] != best_a:
                policy_stable = False
            policy[s] = best_a

        if policy_stable:
            return policy, value
