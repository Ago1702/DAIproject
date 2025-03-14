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
    
    def forward(self, S:torch.Tensor) -> torch.Tensor:
        outs = S
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
        outs = torch.relu(outs)
        outs = self.output(outs)

        return outs
    
actor_func = ActorNet().to(device)
value_func = ValueNet().to(device)

gamma = 0.99

def pickup_sample(s):
    with torch.no_grad():
        s_batch = np.expand_dims(s, axis=0)
        s_batch = torch.tensor(s_batch, dtype=torch.float).to(device)

        logits = actor_func.forward(s_batch)
        logits = logits.squeeze(dim=0)

        probs = F.softmax(logits, dim=-1)

        a = torch.multinomial(probs, num_samples=1)

        return a.tolist()[0]
    
env = gym.make("CartPole-v1")

reward_records = []

opt_actor = torch.optim.AdamW(actor_func.parameters(), lr=0.001)
opt_value = torch.optim.AdamW(value_func.parameters(), lr=0.001)

for i in range(1500):

    done = False
    states = []
    actions = []
    rewards = []

    s, _ = env.reset()

    while not done:
        states.append(s.tolist())
        a = pickup_sample(s)
        s, r, term, trunc, _ = env.step(a)
        done = term or trunc
        actions.append(a)
        rewards.append(r)
    
    cum_rewards = np.zeros_like(rewards)
    reward_len = len(rewards)

    for j in reversed(range(reward_len)):
        cum_rewards[j] = rewards[j] + (cum_rewards[j + 1] if j + 1 < reward_len else 0)
    
    opt_value.zero_grad()

    states = torch.tensor(states, dtype=torch.float).to(device)
    cum_rewards = torch.tensor(cum_rewards, dtype=torch.float).to(device)
    values = value_func.forward(states)
    values = values.squeeze(dim=-1)

    vf_loss = F.mse_loss(
        values,
        cum_rewards,
        reduction='none'
    )
    vf_loss.sum().backward()
    opt_value.step()

    with torch.no_grad():
        values = value_func(states)
    
    opt_actor.zero_grad()

    actions = torch.tensor(actions, dtype=torch.int64).to(device)
    advantages = cum_rewards - values

    logits = actor_func(states)
    log_probs = -F.cross_entropy(logits, actions, reduction='none')
    pi_loss = -log_probs * advantages
    pi_loss.sum().backward()
    opt_actor.step()

    print(f"Run episode {i + 1} with rewards {sum(rewards)}", end='\r')
    reward_records.append(sum(rewards))

    if np.average(reward_records[-50:]) > 475.0:
        break

print('\nDone')

env.close()


env = gym.make("CartPole-v1", render_mode='human')
s, _ = env.reset()

with torch.no_grad():
    done = False
    states = []
    actions = []
    rewards = []
    while not done:    
        states.append(s.tolist())
        a = pickup_sample(s)
        s, r, term, trunc, _ = env.step(a)
        done = term or trunc
        actions.append(a)
        rewards.append(r)