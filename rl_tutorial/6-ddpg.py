import random
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

from util.cartpole import CartPole

env = CartPole()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class QNet(nn.Module):
    def __init__(self, hidden_dim:int=64):
        super(QNet, self).__init__()

        self.hidden = nn.Linear(5, hidden_dim)
        self.output = nn.Linear(hidden_dim, 1)
    
    def forward(self, s, a):
        outs = torch.concat((s, a), dim=-1)
        outs = self.hidden(outs)
        outs = F.relu(outs)
        outs = self.output(outs)
        return outs

q_origin_model = QNet().to(device)
q_target_model = QNet().to(device)
_ = q_target_model.requires_grad_(False)

class PolicyNet(nn.Module):
    def __init__(self, hidden_dim:int=64):
        super(PolicyNet, self).__init__()

        self.hidden = nn.Linear(4, hidden_dim)
        self.output = nn.Linear(hidden_dim, 1)

    def forward(self, s):
        outs = self.hidden(s)
        outs = F.relu(outs)
        outs = self.output(outs)
        outs = torch.tanh(outs)
        return outs
    
mu_origin_model = PolicyNet().to(device)
mu_target_model = PolicyNet().to(device)

_ = mu_target_model.requires_grad_(False)

gamma = 0.99
opt_q = torch.optim.AdamW(q_origin_model.parameters(), lr=0.005)
opt_mu = torch.optim.AdamW(mu_origin_model.parameters(), lr=0.005)

def optimize(states, actions, rewards, next_states, dones):
    states = torch.tensor(states, dtype=torch.float).to(device)
    actions = torch.tensor(actions, dtype=torch.float).to(device)
    actions = actions.unsqueeze(1)
    rewards = torch.tensor(rewards, dtype=torch.float).to(device)
    rewards = rewards.unsqueeze(1)
    next_states = torch.tensor(next_states, dtype=torch.float).to(device)
    dones = torch.tensor(dones, dtype=torch.float).to(device)
    dones = dones.unsqueeze(1)

    opt_q.zero_grad()
    q_org = q_origin_model(states, actions)
    mu_tgt_next = mu_target_model(next_states)
    q_tgt_next = q_target_model(next_states, mu_tgt_next)
    q_tgt = rewards + gamma * (1 - dones) * q_tgt_next
    loss_q = F.mse_loss(
        q_org,
        q_tgt,
        reduction='none'
    )
    loss_q.sum().backward()
    opt_q.step()

    opt_mu.zero_grad()
    mu_org = mu_origin_model(states)
    for p in q_origin_model.parameters():
        p.requires_grad = False
    q_tgt_max = q_origin_model(states, mu_org)
    (-q_tgt_max).sum().backward()
    opt_mu.step()
    for p in q_origin_model.parameters():
        p.requires_grad = True


tau = 0.002

def update_target():
    for var, var_target in zip(q_origin_model.parameters(), q_target_model.parameters()):
        var_target.data = tau * var.data + (1.0 - tau) * var_target.data
    for var, var_target in zip(mu_origin_model.parameters(), mu_target_model.parameters()):
        var_target.data = tau * var.data + (1.0 - tau) * var_target.data
    

class ReplayBuffer:
    def __init__(self, buffer_size:int):
        self.buffer_size = buffer_size
        self.buffer = []
    
    def add(self, item):
        if len(self.buffer) == self.buffer_size:
            self.buffer.pop(0)
        self.buffer.append(item)
    
    def sample(self, batch_size:int):
        items = random.sample(self.buffer, batch_size)
        states = [i[0] for i in items]
        actions = [i[1] for i in items]
        rewards = [i[2] for i in items]
        n_states = [i[3] for i in items]
        dones = [i[4] for i in items]

        return states, actions, rewards, n_states, dones

    def length(self):
        return len(self.buffer)
    
buffer = ReplayBuffer(20000)

class OrnsteinUhlenbeckActionNoise:
    def __init__(self, mu, sigma, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()
    
    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)
    
    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x
    
ou_action_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(1), sigma=np.ones(1) * 0.05)

def pick_sample(s):
    with torch.no_grad():
        s = np.array(s)
        s_batch = np.expand_dims(s, axis=0)
        s_batch = torch.tensor(s_batch, dtype=torch.float).to(device)
        action_det = mu_origin_model(s_batch)
        action_det = action_det.squeeze(dim=1)
        noise = ou_action_noise()
        action = action_det.cpu().numpy() + noise
        action = np.clip(action, -1.0, 1.0)
        return float(action.item())


batch_size = 250

reward_records = []
mean = 0
for i in range(10000):
    s = env.reset()
    done = False
    cum_reward = 0
    while not done:
        a = pick_sample(s)
        s_next, r, term, trunc, _ = env.step(a)
        done = term or trunc
        buffer.add([s, a, r, s_next, float(term)])
        cum_reward += r
        if buffer.length() >= batch_size:
            states, actions, rewards, n_states, dones = buffer.sample(batch_size)
            optimize(states, actions, rewards, n_states, dones)
            update_target()
        s = s_next
    reward_records.append(cum_reward)
    if (i + 1) % 50 == 0:
        mean = np.average(reward_records[-50:])
    print("Run episode{} with rewards {}, mean {}".format(i, cum_reward, mean), end="\r")
    
    if np.average(reward_records[-50:]) >= 475.0:
        break

    