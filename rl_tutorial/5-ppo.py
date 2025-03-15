import gymnasium as gym

import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ActorNet(nn.Module):
    def __init__(self, hidden_dim:int=16):
        super(ActorNet, self).__init__()

        self.hidden = nn.Linear(4, hidden_dim)
        self.output = nn.Linear(hidden_dim, 2)

    def forward(self, s:torch.Tensor) -> torch.Tensor:
        outs = s
        outs = self.hidden(outs)
        outs = F.relu(outs)
        outs = self.output(outs)
        return outs

class ValueNet(nn.Module):
    def __init__(self, hidden_dim:int=16):
        super(ValueNet, self).__init__()

        self.hidden = nn.Linear(4, hidden_dim)
        self.output = nn.Linear(hidden_dim, 1)

    def forward(self, s:torch.Tensor) -> torch.Tensor:
        outs = s
        outs = self.hidden(outs)
        outs = F.relu(outs)
        outs = self.output(outs)
        return outs

actor_func = ActorNet().to(device)
value_func = ValueNet().to(device)

gamma = 0.99

kl_coeff = 0.20
vf_coeff = 0.50

def pick_sample_and_logp(s):
    with torch.no_grad():
        s_batch = np.expand_dims(s, axis=0)
        s_batch = torch.tensor(s_batch, dtype=torch.float).to(device)
        logits = actor_func(s_batch).squeeze(0)
        probs = F.softmax(logits, -1)
        a = torch.multinomial(probs, 1)
        a.squeeze_(0)
        logprb = -F.cross_entropy(logits, a, reduction='none')

        return a.tolist(), logits.tolist(), logprb.tolist()

env = gym.make('CartPole-v1')
reward_records = []

all_params = list(actor_func.parameters()) + list(value_func.parameters())
opt = torch.optim.AdamW(all_params, lr=0.0005)

for i in range(5000):
    done = False
    states = []
    actions = []
    logits = []
    logprbs = []
    rewards = []
    s, _ = env.reset()
    while not done:
        states.append(s)
        a, l, p = pick_sample_and_logp(s)
        s, r, term, trunc, _ = env.step(a)
        done = term or trunc
        actions.append(a)
        logits.append(l)
        logprbs.append(p)
        rewards.append(r)
    
    cum_rewards = np.zeros_like(rewards)
    reward_len = len(rewards)
    for j in reversed(range(reward_len)):
        cum_rewards[j] = rewards[j] + (cum_rewards[j + 1] if j+1 < reward_len else 0)

    opt.zero_grad()
    states = torch.tensor(states, dtype=torch.float).to(device)
    actions = torch.tensor(actions, dtype=torch.int64).to(device)
    logits_old = torch.tensor(logits, dtype=torch.float).to(device)
    logprbs = torch.tensor(logprbs, dtype=torch.float).to(device)
    logprbs = logprbs.unsqueeze(1)
    cum_rewards = torch.tensor(cum_rewards, dtype=torch.float).to(device)
    cum_rewards = cum_rewards.unsqueeze(1)

    values_new = value_func(states)
    logits_new = actor_func(states)

    advantages = cum_rewards - values_new

    advantages = (advantages - advantages.mean()) / advantages.std()

    logprbs_new = -F.cross_entropy(logits_new, actions, reduction='none')
    logprbs_new = logprbs_new.unsqueeze(1)

    prob_ratio = torch.exp(logprbs_new - logprbs)

    l0 = logits_old - torch.amax(logits_old, dim=1, keepdim=True)
    l1 = logits_old - torch.amax(logits_old, dim=1, keepdim=True)

    e0 = torch.exp(l0)
    e1 = torch.exp(l1)

    e_sum0 = torch.sum(e0, dim=1, keepdim=True)
    e_sum1 = torch.sum(e1, dim=1, keepdim=True)

    p0 = e0 / e_sum0

    kl = torch.sum(
        p0 * (l0 - torch.log(e_sum0) - l1 + torch.log(e_sum1)),
        dim=1,
        keepdim=True
    )

    vf_loss = F.mse_loss(
        values_new,
        cum_rewards, 
        reduction='none'
    )

    loss = -advantages * prob_ratio + kl + kl_coeff + vf_loss * vf_coeff

    loss.sum().backward()

    opt.step()

    mean = np.average(reward_records[-200:])
    print("Run episode {} with rewards {}, Mean {}".format(i + 1, np.sum(rewards), mean), end="\r")
    reward_records.append(np.sum(rewards))

    # stop if reward mean > 475.0
    if mean > 475.0:
        break

print('\nDone')
env.close()

env = gym.make('CartPole-v1', render_mode='human')

with torch.no_grad():
    done = False
    states = []
    actions = []
    logits = []
    logprbs = []
    rewards = []
    s, _ = env.reset()
    while not done:
        states.append(s)
        a, l, p = pick_sample_and_logp(s)
        s, r, term, trunc, _ = env.step(a)
        done = term or trunc
        actions.append(a)
        logits.append(l)
        logprbs.append(p)
        rewards.append(r)