import numpy as np
from grid_world.dynamics import *

BOARD_ROWS = 3
BOARD_COLS = 4
WIN_STATE = (0, 3)
LOSE_STATE = (1, 3)
START = (2, 0)
DETERMINISTIC = False

def run_grid_world(policy, state_value, action_value, rounds=100):
    print("Let's run grid world!")
    agent = Agent(policy)
    agent.play(rounds)
    agent.show_policy()
    print("\nstate value:\n {}".format(np.round(state_value, 3)))
    print("action value:\n {}".format(np.round(action_value, 3)))

class environment:
    def __init__(self, state=START):
        self.board = np.zeros([BOARD_ROWS, BOARD_COLS])
        self.board[1, 1] = -1
        self.state = state
        self.isEnd = False
        self.determine = DETERMINISTIC

    def giveReward(self):
        if self.state == WIN_STATE:
            return 1
        elif self.state == LOSE_STATE:
            return -1
        else:
            return 0

    def isEndFunc(self):
        if (self.state == WIN_STATE) or (self.state == LOSE_STATE):
            self.isEnd = True

    def _chooseActionProb(self, action):
        if action == "up":
            return np.random.choice(["up", "left", "right"], p=[0.8, 0.1, 0.1])
        if action == "down":
            return np.random.choice(["down", "left", "right"], p=[1.0, 0.0, 0.0])
        if action == "left":
            return np.random.choice(["left", "up", "down"], p=[1.0, 0.0, 0.0])
        if action == "right":
            return np.random.choice(["right", "up", "down"], p=[1.0, 0.0, 0.0])

    def nextPosition(self, action):
        """
        action: up, down, left, right
        -------------
        0 | 1 | 2| 3|
        1 |
        2 |
        return next position on board
        """
        if self.determine:
            if action == "up":
                nextState = (self.state[0] - 1, self.state[1])
            elif action == "down":
                nextState = (self.state[0] + 1, self.state[1])
            elif action == "left":
                nextState = (self.state[0], self.state[1] - 1)
            else:
                nextState = (self.state[0], self.state[1] + 1)
            self.determine = False
        else:
            # non-deterministic
            action = self._chooseActionProb(action)
            self.determine = True
            nextState = self.nextPosition(action)

        # if next state is legal
        if (nextState[0] >= 0) and (nextState[0] <= 2):
            if (nextState[1] >= 0) and (nextState[1] <= 3):
                if nextState != (1, 1):
                    return nextState
        return self.state

    def showBoard(self):
        self.board[self.state] = 1
        for i in range(0, BOARD_ROWS):
            print('-----------------')
            out = '| '
            for j in range(0, BOARD_COLS):
                if self.board[i, j] == 1:
                    token = '*'
                if self.board[i, j] == -1:
                    token = 'z'
                if self.board[i, j] == 0:
                    token = '0'
                out += token + ' | '
            print(out)
        print('-----------------')
    
    def show_action_board(self, action):
        
        for i in range(0, BOARD_ROWS):
            print('-----------------')
            out = '| '
            for j in range(0, BOARD_COLS):
                if 4*i + j == 3:
                    token = 'G'
                if 4*i + j == 7:
                    token = 'T'
                if self.board[i, j] == -1:
                    token = 'z'
                else:
                    token = action[4*i + j]
                out += token + ' | '
            print(out)
        print('-----------------')

class randomAgent:
    def __init__(self):
        self.states = []  # record position and action taken at the position
        self.actions = ["up", "down", "left", "right"]
        self.env = environment()
        self.isEnd = self.env.isEnd
        self.result_stat = []

    def randomAction(self):
        action = np.random.choice(self.actions)
        return action

    def takeAction(self, action):
        position = self.env.nextPosition(action)
        return environment(state=position)

    def reset(self):
        self.states = []
        self.env = environment()
        self.isEnd = self.env.isEnd

    def play(self, rounds):
        i=0
        self.result_stat = []
        while i < rounds:
            if self.env.isEnd:
                reward = self.env.giveReward()
                if reward >= 1:
                    self.result_stat.append(reward)
                self.reset()
                i += 1
            else:
                action = self.randomAction()
                self.states.append([(self.env.state), action])
                self.env = self.takeAction(action)
                self.env.isEndFunc()
                self.isEnd = self.env.isEnd
        success_rate = np.sum(self.result_stat) / rounds
        print("Success rate:{} %".format(success_rate * 100))

class Agent:
    def __init__(self, policy):
        self.states = []  # record position and action taken at the position
        self.actions = ["up", "left", "right", "down"]
        self.env = environment()
        self.isEnd = self.env.isEnd
        self.result_stat = []
        self.policy = policy

    def Action(self, state):
        # action = np.random.choice(self.actions)
        state = reverse_position(state)
        action = np.random.choice(self.actions, p=self.policy[state,:])
        return action

    def takeAction(self, action):
        position = self.env.nextPosition(action)
        return environment(state=position)

    def reset(self):
        self.states = []
        self.env = environment()
        self.isEnd = self.env.isEnd

    def play(self, rounds):
        i=0
        self.result_stat = []
        while i < rounds:
            if self.env.isEnd:
                reward = self.env.giveReward()
                if reward >= 1:
                    self.result_stat.append(reward)
                self.reset()
                i += 1
            else:
                action = self.Action(self.env.state)
                self.states.append([(self.env.state), action])
                self.env = self.takeAction(action)
                self.env.isEndFunc()
                self.isEnd = self.env.isEnd
        success_rate = np.sum(self.result_stat) / rounds
        print("Success rate:{} %".format(success_rate * 100))
    
    def show_policy(self):
        action = policy_to_action(self.policy)
        self.env.show_action_board(action)

# dynamics module
class dynamics:
    def __init__(self, pi, gamma, reward):
        self.pi = pi    
        self.gamma = gamma
        self.reward = reward
        # state, next_state, action, reward
        self.dynamics = np.zeros((12, 12, 4, 3)) 
        self.dynamics = self.return_dynamics()
        # w/o tensor notation
        self.pi_dynamics = np.zeros_like(self.dynamics) 
        self.P_reward = np.zeros((12, 3))
        self.P_value = np.zeros((12, 12))
        self.pi_dynamics, self.P_reward, self.P_value = self.update_return_all()
        
    def return_dynamics(self):
        for i in range(12):
            # press 'up'
            if compute_position(i)[0] == 0: # top row
                self.dynamics[i, i, 0, 1] += 0.8
            elif i == 9: # wall
                self.dynamics[i, i, 0, 1] += 0.8
            else:
                if i not in [3, 5, 7]:
                    position = compute_position(i) - np.array([1, 0])
                    self.dynamics[i, reverse_position(position), 0, 1] += 0.8
            # press 'left'
            if compute_position(i)[1] == 0: # top left
                self.dynamics[i, i, 0, 1] += 0.1
                self.dynamics[i, i, 1, 1] += 1
            elif i == 6: # wall
                self.dynamics[i, i, 0, 1] += 0.1
                self.dynamics[i, i, 1, 1] += 1
            else:
                if i not in [3, 5, 7]:
                    position = compute_position(i) - np.array([0, 1])
                    self.dynamics[i, reverse_position(position), 0, 1] += 0.1
                    self.dynamics[i, reverse_position(position), 1, 1] += 1
            # press 'right'
            if compute_position(i)[1] == 3: # top-right column
                self.dynamics[i, i, 0, 1] += 0.1
                self.dynamics[i, i, 2, 1] += 1
            elif i == 4: # wall
                self.dynamics[i, i, 0, 1] += 0.1
                self.dynamics[i, i, 2, 1] += 1
            else:
                position = compute_position(i) + np.array([0, 1])
                if i not in [3, 5, 7]:
                    self.dynamics[i, reverse_position(position), 0, 1] += 0.1
                    self.dynamics[i, reverse_position(position), 2, 1] += 1
            # press 'down'
            if compute_position(i)[0] == 2: # buttom row
                self.dynamics[i, i, 3, 1] += 1
            elif i == 1: # wall
                self.dynamics[i, i, 3 ,1] += 1
            else:
                if i not in [3, 5, 7]:
                    position = compute_position(i) + np.array([1, 0])
                    self.dynamics[i, reverse_position(position), 3, 1] += 1
            if i in [3, 5, 7]: # end, wall cases
                self.dynamics[i, :, :, :] = 0 # absorbing state
                if i == 3:
                    self.dynamics[i, 8, :, 0] = 1 # goal state
                if i == 5:
                    self.dynamics[i, i, :, 1] = 1 # wall
                if i == 7:
                    self.dynamics[i, 8, :, 2] = 1 # trap state
        return self.dynamics            

    def update_pi_dynamics(self, pi_dynamics):
        for i in range(12):
            for a in range(4):
                # broadcasting
                pi_dynamics[i, :, a, :] = self.pi[i, a] * self.dynamics[i, :, a, :]
        return pi_dynamics

    def update_reward(self, P_reward):
        # state -> reward
        for j in range(12):
             for r in range(3):
                 # marginalization
                 P_reward[j, r] = np.sum(self.pi_dynamics[j, :, :, r])
        return P_reward

    def update_value(self, P_value):
        # state -> state
        for j in range(12):
             for i in range(12):
                 # marginalization
                 P_value[j, i] = np.sum(self.pi_dynamics[j, i, :, :])
        return P_value

    def update_return_all(self):
        return self.update_pi_dynamics(self.pi_dynamics), self.update_reward(self.P_reward), self.update_value(self.P_value)

    def compute_state_value(self):
        coeff = np.eye(12) - self.gamma * self.P_value
        inv_coeff = np.linalg.inv(coeff)
        state_value = inv_coeff @ self.P_reward @ self.reward
        return state_value

    def compute_action_value(self):
        state_value = self.compute_state_value()
        expectation_reward = np.zeros((12, 4))
        expectation_value = np.zeros((12, 4))
        for i in range(12):
            for a in range(4):
                expectation_reward[i, a] = self.reward @ np.sum(self.dynamics, axis=1)[i, a, :]
                expectation_value[i, a] = self.gamma * state_value @ np.sum(self.dynamics, axis=3)[i, :, a]
        action_value = expectation_reward + expectation_value
        return action_value