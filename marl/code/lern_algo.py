import torch

import numpy as np
import torch.nn as nn
import marl.code.utils as utils

from torch.optim import Adam
from marl.code.model import MultiHeadActor, QNetwork

class MultiTD3(object):
    '''
    Class MultiTD3: 
    '''

    def __init__(self, id, algo_name, state_dim:int, action_dim:int, hidden_size:int,
                    actor_lr, critic_lr, gamma, tau, save_tag, folder, gpu:bool, num_agents, init_w=True):
        self.id = id
        self.algo_name = algo_name
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_size = hidden_size
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.gamma = gamma
        self.tau = tau
        self.save_tag = save_tag
        self.folder = folder
        self.gpu = gpu
        self.num_agents = num_agents
        self.init_w = init_w

        self.total_update = 0

        self.policy = MultiHeadActor(state_dim, action_dim, hidden_size, num_agents)
        if init_w:
            self.policy.apply(utils.init_weights)
        self.policy_target = MultiHeadActor(state_dim, action_dim, hidden_size, num_agents)
        utils.hard_update(self.policy_target, self.policy)
        self.policy_optim = Adam(self.policy.parameters(), self.actor_lr)

        self.critic = QNetwork(state_dim, action_dim, hidden_size)
        if init_w:
            self.critic.apply(utils.init_weights)
        self.critic_target = QNetwork(state_dim, action_dim, hidden_size)
        utils.hard_update(self.critic_target, self.critic)
        self.critic_optim = Adam(self.critic.parameters(), self.critic_lr)

        self.loss = nn.MSELoss()

        if gpu:
            self.policy_target.cuda()
            self.critic_target.cuda()
            self.policy.cuda()
            self.policy.cuda()
        
        self.num_critic_updates = 0

        self.policy_loss = {'min': None, 'max': None, 'mean': None, 'std': None}
        self.q_loss = {'min': None, 'max': None, 'mean': None, 'std': None}
        self.q = {'min': None, 'max': None, 'mean': None, 'std': None}

    def update_param(self, state_batch, n_state_batch,
                     action_batch, reward_batch, done_batch,
                     g_reward, id_agent:int, num_epoch:int = 1, **kwargs):
        
        if isinstance(state_batch, list):
            state_batch = torch.cat(state_batch)
            n_state_batch = torch.cat(n_state_batch)
            action_batch = torch.cat(action_batch)
            reward_batch = torch.cat(reward_batch)
            done_batch = torch.cat(done_batch)
            g_reward = torch.cat(g_reward)
        
        for _ in range(num_epoch):
            with torch.no_grad():
                policy_noise = np.random.normal(0, kwargs['policy_noise'], (action_batch.size()[0], action_batch.size()[1]))
                policy_noise = torch.clamp(policy_noise, -kwargs['policy_noise_clip'], kwargs['policy_noise_clip'])

                n_action_batch = self.policy_target.action(n_state_batch, id_agent) + policy_noise.cuda() if self.gpu else policy_noise

                n_action_batch = torch.clamp(n_action_batch, -1, +1)

                q1, q2 = self.critic_target.forward(n_state_batch, n_action_batch)
                q1 = (1 - done_batch) * q1
                q2 = (1 - done_batch) * q2

                if self.algo_name == 'TD3':
                    next_q = torch.min(q1, q2)
                elif self.algo_name == 'DDPG':
                    next_q = q1
                
                target_q = reward_batch + (self.gamma * next_q)
            
            self.critic_optim.zero_grad()
            current_q1, current_q2 = self.critic.forward(state_batch, action_batch)

            dt = self.loss(current_q1, target_q)

            if self.algo_name == "TD3":
                dt = dt + self.loss(current_q2, target_q)
            
            dt.backward()
            self.critic_optim.step()
            self.num_critic_updates += 1

            if self.num_critic_updates % kwargs["policy_ups_freq"] == 0:

                actor_actions = self.policy.action(state_batch, id_agent)

                Q1, Q2 = self.critic.forward(state_batch, actor_actions)

                policy_loss = -Q1

                policy_loss = policy_loss.mean()

                self.policy_optim.zero_grad()

                policy_loss.backward(retain_graph=True)
                self.policy_optim.step()
            
            if self.num_critic_updates % kwargs["policy_ups_freq"] == 0:
                utils.soft_update(self.policy_target, self.policy, self.tau)
            
            utils.soft_update(self.critic_target, self.critic, self.tau)

            self.total_update += 1



class MATD3(object):
    '''
    Class MATD3: 
    '''

    def __init__(self, id, algo_name, state_dim, action_dim, hidden_size,
                 actor_lr, critic_lr, gamma, tau, save_tag, folder, gpu,
                 num_agents, init_w:bool = True):
        self.id = id
        self.algo_name = algo_name
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_size = hidden_size
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.gamma = gamma
        self.tau = tau
        self.save_tag = save_tag
        self.folder_name = folder
        self.gpu = gpu
        self.num_agents = num_agents
        self.init_w = init_w
        self.total_update = 0

        self.policy = MultiHeadActor(state_dim, action_dim, hidden_size, num_agents)
        if init_w:
            self.policy.apply(utils.init_weights)
        self.policy_target = MultiHeadActor(state_dim, action_dim, hidden_size, num_agents)
        utils.hard_update(self.policy_target, self.policy)
        self.policy_optim = Adam(self.policy.parameters(), actor_lr)

        self.critics = [QNetwork(state_dim * num_agents, action_dim*num_agents, hidden_size * 3) for _ in range(self.num_agents)]
        self.critics_target = [QNetwork(state_dim * num_agents, action_dim*num_agents, hidden_size * 3) for _ in range(self.num_agents)]
        if init_w:
            for critic, critic_target in zip(self.critics, self.critics_target):
                critic.apply(utils.init_weights)
                utils.hard_update(critic_target, critic)
        self.critics_optims = [Adam(critic.parameters(), critic_lr) for critic in self.critics]

        self.loss = nn.MSELoss()

        if gpu:
            self.policy_target.cuda()
            self.policy.cuda()
            for critic, critic_target in zip(self.critics, self.critics_target):
                critic.cuda()
                critic_target.cuda()
        
        self.num_critic_updates = 0
        self.policy_loss = {'min':None, 'max': None, 'mean':None, 'std':None}
        self.q_loss = {'min':None, 'max': None, 'mean':None, 'std':None}
        self.q = {'min':None, 'max': None, 'mean':None, 'std':None}
