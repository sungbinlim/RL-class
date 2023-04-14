import numpy as np

def compute_Q_reward(reward, dynamics):
    Q_reward = np.zeros((12, 4))
    for a in range(4):
        for i in range(12):
            Q_reward[i, a] = np.sum(dynamics[i, :, a, :] @ reward)
    return Q_reward

def compute_Q_value(action_value, dynamics, gamma):
    Q_value = np.zeros((12, 4))
    for i in range(12):
            for a in range(4):
                Q_value[i, a] = gamma * np.max(action_value, 1) @ np.sum(dynamics, axis=3)[i, :, a]
    return Q_value

def update_value_iteration(Q_value, Q_reward):
    
    action_value = Q_reward + Q_value
    return action_value

# value iteration
def value_iteration(init_action_value, gamma, reward, dynamics, eps=1e-8):

    init_dynamics = dynamics() # state, next_state, action, reward
    init_dynamics = init_dynamics.dynamics
    action_value = init_action_value # state, action
    
    Q_reward = compute_Q_reward(reward, init_dynamics)

    advances = np.inf
    n_it = 0

    while advances > eps or n_it <= 3:

        Q_value = compute_Q_value(action_value, init_dynamics, gamma)
        new_action_value = update_value_iteration(Q_value, Q_reward)
        advances = np.sum(np.abs(new_action_value - action_value))
        action_value = new_action_value
         
        n_it += 1

    print("Value iteration converged. (iteration={}, eps={})".format(n_it, np.sum(advances)))

    return new_action_value