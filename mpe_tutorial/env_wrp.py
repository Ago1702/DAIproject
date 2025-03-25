from pettingzoo.mpe import simple_spread_v3
import pettingzoo as ptz
from model import ActorNet, QNet

import numpy as np

import torch
from torch import nn as nn
from torch.nn import functional as F

from torchrl.data import ReplayBuffer, ListStorage

from copy import deepcopy

class EnvWrapper:
    '''
    Class EnvWrapper: Un gestore esterno pper semplificare la gestione dell'environment 
                        simple-spread

    '''

    def __init__(self, env_type=simple_spread_v3, kwargs:dict=None):
        if kwargs is None:
            self.kwargs = {'continuous_actions':True}
        else:
            self.kwargs = kwargs
        

        self.env_type = env_type
        self.render_mode = None
        self.env:ptz.AECEnv = self.env_type.env(**self.kwargs)
        
        self.env.reset()
        self.agents = deepcopy(self.env.agents)
        self.cum_rew = np.array([0.] * len(self.agents))
        self.print_rewards = False
    


    def run_episode(self, agent, model:ActorNet = None):
        observation, reward, termination, truncation, info = self.env.last()

        done = termination or truncation

        if done:
            action = None
        elif model is None:
            action = self.env.action_space(agent).sample()
        else:
            action = model.noisy_forward(torch.tensor(observation)).numpy().squeeze(0)

        self.env.step(action)

        return observation, action, reward, done
    
    def render_round(self, render_mode:str='human', model = None):
        self.env = self.env_type.env(render_mode=render_mode, **self.kwargs)
        self.env.reset()
        for agent in self.env.agent_iter():
            if isinstance(model, dict):
                self.run_episode(agent, model[agent])
            else:
                self.run_episode(agent, model)
        
        self.env = self.env_type.env(**self.kwargs)
        self.env.reset()

    def gather_episode(self, model):
        state = self.env.state().tolist()
        observations = [None] * len(self.agents)
        actions = [None] * len(self.agents)
        dones = [None] * len(self.agents)
        new_state = None
        new_observations = None
        rewards = None


        for agent in self.agents:
            index = self.agents.index(agent)
            if isinstance(model, dict):
                observations[index], actions[index], _, dones[index] = self.run_episode(agent, model[agent])
            else:
                observations[index], actions[index], _, dones[index] = self.run_episode(agent, model)
        
        if not any(dones):
            new_state = self.env.state().tolist()
            new_observations = [self.env.observe(agent).tolist() for agent in self.agents]
            rewards = [self.env._cumulative_rewards[agent] for agent in self.agents]
        
        return dones, state, observations, actions, rewards, new_state, new_observations
        
    def fill_buffer(self, buffer:ReplayBuffer, model:nn.Module|dict[str, nn.Module]=None, num_episodes:int=100, device='cpu'):
        rewards_history = []
        for i in range(num_episodes):
            dones, state, observations, actions, rewards, new_state, new_observations = self.gather_episode(model)
            if any(dones):
                self.env.reset()
                self.cum_rew.fill(0.)
                dones, state, observations, actions, rewards, new_state, new_observations = self.gather_episode(model)
            self.cum_rew = self.cum_rew + rewards
            rewards_history.append(rewards)
            observations = [obs.tolist() for obs in observations]
            actions = [act.tolist() for act in actions]
            buffer.add(
                (torch.tensor(state).to(device), torch.tensor(observations).to(device), torch.tensor(actions).to(device),
                 torch.tensor(rewards).to(device), torch.tensor(new_state).to(device), torch.tensor(new_observations).to(device), torch.tensor(self.cum_rew.tolist()).to(device))
            )
        return rewards_history


'''if __name__ == '__main__':
    env = EnvWrapper()
    #env.render_round()
    print(env.cum_rew)

    buffer = ReplayBuffer(storage=ListStorage(1_000_000))

    env.fill_buffer(None, buffer, 1000)

    state, observation, actions, rewards, new_state, new_obseravtions, cum_reward = buffer[:10]
    print(cum_reward, '\n\n', rewards)'''


