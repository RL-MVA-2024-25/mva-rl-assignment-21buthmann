import os
import random
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from gymnasium.wrappers import TimeLimit

from env_hiv import HIVPatient
from evaluate import evaluate_HIV

env = TimeLimit(
    env=HIVPatient(domain_randomization=True), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.


# You have to implement your own agent.
# Don't modify the methods names and signatures, but you can add methods.
# ENJOY!
class ProjectAgent:
    def act(self, observation, use_random=False):
        with torch.no_grad():
            Q = self.model(torch.Tensor(observation).unsqueeze(0).to("cpu"))
            return torch.argmax(Q).item()

    def save(self, path):
        self.path = path + "/dqn_model.pt"
        torch.save(self.model.state_dict(), self.path)
        return

    def load(self):
        device = torch.device('cpu')
        self.path = os.getcwd() + "/dqn_model.pt"
        self.model = self.myDQN({}, device)
        self.model.load_state_dict(torch.load(self.path, map_location=device))
        self.model.eval()
        return

    def act_greedy(self, DQN, state):
        with torch.no_grad():
            Q = DQN(torch.Tensor(state).unsqueeze(0).to("cpu"))
            return torch.argmax(Q).item()
        
    def gradient_step(self):
        if len(self.memory) > self.batch_size:
            X, A, R, Y, D = self.memory.sample(self.batch_size)
            QYmax = self.target_model(Y).max(1)[0].detach()
            update = torch.addcmul(R, 1-D, QYmax, value=self.gamma)
            QXA = self.model(X).gather(1, A.to(torch.long).unsqueeze(1))
            loss = self.criterion(QXA, update.unsqueeze(1))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step() 

    def myDQN(self, config, device):

        state_dim = env.observation_space.shape[0]
        n_action = env.action_space.n
        nb_neurons=256

        DQN = torch.nn.Sequential(
            nn.Linear(state_dim, nb_neurons),
            nn.ReLU(),
            nn.Linear(nb_neurons, nb_neurons),
            nn.ReLU(),
            nn.Linear(nb_neurons, nb_neurons),
            nn.ReLU(),
            nn.Linear(nb_neurons, nb_neurons),
            nn.ReLU(),
            nn.Linear(nb_neurons, n_action)
            ).to(device)
        return DQN

    def train(self):
        config = {'nb_actions': env.action_space.n,
                'learning_rate': 0.001,
                'gamma': 0.98,
                'buffer_size': 100000,
                'epsilon_min': 0.02,
                'epsilon_max': 1.,
                'epsilon_decay_period': 21000, 
                'epsilon_delay_decay': 100,
                'batch_size': 790,
                'gradient_steps': 3,
                'update_target_freq': 400,
                'criterion': torch.nn.SmoothL1Loss()}

    
        device = torch.device('cpu')
        print('Using device:', device)
        self.model = self.myDQN(config, device)
        self.target_model = deepcopy(self.model).to(device)

        self.gamma = config['gamma']
        self.batch_size = config['batch_size']
        self.nb_actions = config['nb_actions']

        epsilon_max = config['epsilon_max']
        epsilon_min = config['epsilon_min']
        epsilon_stop = config['epsilon_decay_period'] if 'epsilon_decay_period' in config.keys() else 1000
        epsilon_delay = config['epsilon_delay_decay'] if 'epsilon_delay_decay' in config.keys() else 20
        epsilon_step = (epsilon_max-epsilon_min)/epsilon_stop

        self.memory = ReplayBuffer(config['buffer_size'], device)

        self.criterion = config['criterion'] if 'criterion' in config.keys() else torch.nn.MSELoss()
        lr = config['learning_rate'] if 'learning_rate' in config.keys() else 0.001
        
        self.optimizer = config['optimizer'] if 'optimizer' in config.keys() else torch.optim.Adam(self.model.parameters(), lr=lr)
        self.optimizer2 = config['optimizer'] if 'optimizer' in config.keys() else torch.optim.Adam(self.model.parameters(), lr=lr)

        nb_gradient_steps = config['gradient_steps'] if 'gradient_steps' in config.keys() else 1

        update_target_freq = config['update_target_freq'] if 'update_target_freq' in config.keys() else 20


        previous_val = 0

        max_episode = 280 

        episode_return = []
        episode = 0
        episode_cum_reward = 0
        state, _ = env.reset()
        epsilon = epsilon_max
        step = 0

        while episode < max_episode:
            if step > epsilon_delay:
                epsilon = max(epsilon_min, epsilon-epsilon_step)
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = self.act_greedy(self.model, state)
            
            next_state, reward, done, trunc, _ = env.step(action)
            self.memory.append(state, action, reward, next_state, done)
            episode_cum_reward += reward
            
            for _ in range(nb_gradient_steps): 
                self.gradient_step()

            if step % update_target_freq == 0: 
                self.target_model.load_state_dict(self.model.state_dict())
            
            step += 1
            if done or trunc:
                episode += 1
                
                validation_score = evaluate_HIV(agent=self, nb_episode=1)
                                
                print(f"Episode {episode:3d} | "
                      f"Epsilon {epsilon:6.2f} | "
                      f"Batch Size {len(self.memory):5d} | "
                      f"Episode Return {episode_cum_reward:.2e} | "
                      f"Evaluation Score {validation_score:.2e}")
                state, _ = env.reset()

                if validation_score > previous_val:
                    previous_val = validation_score
                    self.best_model = deepcopy(self.model).to(device)
                    path = os.getcwd()
                    self.save(path)
                episode_return.append(episode_cum_reward)
                
                episode_cum_reward = 0
            else:
                state = next_state

        self.model.load_state_dict(self.best_model.state_dict())
        path = os.getcwd()
        self.save(path)
        return episode_return


class ReplayBuffer:
    def __init__(self, capacity, device):
        self.capacity = capacity 
        self.data = []
        self.index = 0
        self.device = device
    def append(self, s, a, r, s_, d):
        if len(self.data) < self.capacity:
            self.data.append(None)
        self.data[self.index] = (s, a, r, s_, d)
        self.index = (self.index + 1) % self.capacity
    def sample(self, batch_size):
        batch = random.sample(self.data, batch_size)
        return list(map(lambda x:torch.Tensor(np.array(x)).to(self.device), list(zip(*batch))))
    def __len__(self):
        return len(self.data)