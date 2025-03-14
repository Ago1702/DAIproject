import gymnasium as gym

import numpy as np

import torch
import torch.nn as nn

from torch.nn import functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class PolicyPi(nn.Module):
    def __init__(self, hidden_dim:int = 64):
        super().__init__()

        self.hidden = nn.Linear(4, hidden_dim)
        self.classify = nn.Linear(hidden_dim, 2)

    def forward(self, S:torch.Tensor) -> torch.Tensor:
        outs = S
        outs = self.hidden(outs)
        outs = F.relu(outs)
        outs = self.classify(outs)
        return outs
    
policy_pi = PolicyPi().to(device)

gamma = 0.99

def pickup_sample(s):
    with torch.no_grad():
        s_batch = np.expand_dims(s, axis=0)
        s_batch = torch.tensor(s_batch, dtype=torch.float).to(device)

        logits = policy_pi.forward(s_batch)

        logits.squeeze_(dim=0)

        probs = F.softmax(logits, dim=-1)

        a = torch.multinomial(probs, num_samples=1)

        return a.tolist()[0]
    
env = gym.make("CartPole-v1")
reward_records = []

opt = torch.optim.AdamW(policy_pi.parameters(), lr=0.001)

for i in range(1250):

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
    rewards_len = len(rewards)
    for j in reversed(range(rewards_len)):
        cum_rewards[j] = rewards[j] + (cum_rewards[j + 1] * gamma if j+1 < rewards_len else 0)

    states = torch.tensor(states, dtype=torch.float).to(device)
    actions = torch.tensor(actions, dtype=torch.int64).to(device)
    cum_rewards = torch.tensor(cum_rewards, dtype=torch.float).to(device)

    opt.zero_grad()

    logits = policy_pi.forward(states)

    log_probs = -F.cross_entropy(logits, actions, reduction='none')
    loss = -log_probs * cum_rewards
    loss.sum().backward()
    opt.step()

    print(f"Run episode {i} with rewards {sum(rewards)}", end='\r')
    reward_records.append (sum(rewards))

print("\nDone")

env = gym.make("CartPole-v1", render_mode='human')


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
        