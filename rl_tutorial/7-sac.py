import random
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

torch.set_default_dtype(torch.float64)

env = gym.make('CartPole-v1')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class PolicyNet(nn.Module):
    def __init__(self, hidden_dim:int = 64):
        super(PolicyNet, self).__init__()

        self.hidden = nn.Linear(4, hidden_dim)
        self.output = nn.Linear(hidden_dim, 2)
    
    def forward(self, s:torch.Tensor) -> torch.Tensor:
        outs = self.hidden(s)
        outs = F.relu(outs)
        outs = self.output(outs)
        return outs

pi_model = PolicyNet().to(device)

def pick_sample(s):
    with torch.no_grad():
        s_batch = np.expand_dims(s, axis=0)
        s_batch = torch.tensor(s_batch, dtype=torch.float64).to(device)

        logits = pi_model.forward(s_batch)

        logits = logits.squeeze(0)

        probs = F.softmax(logits, dim=-1)

        a = torch.multinomial(probs, num_samples=1)

        a = a.squeeze(0)

        return a.tolist()

class QNet(nn.Module):
    def __init__(self, hidden_dim:int = 64):
        super(QNet, self).__init__()

        self.hidden = nn.Linear(4, hidden_dim)
        self.output = nn.Linear(hidden_dim, 2)
    
    def forward(self, s:torch.Tensor) -> torch.Tensor:
        outs = self.hidden(s)
        outs = F.relu(outs)
        outs = self.output(outs)
        return outs

q_origin_model1 = QNet().to(device)
q_origin_model2 = QNet().to(device)
q_target_model1 = QNet().to(device)
q_target_model2 = QNet().to(device)

_ = q_target_model1.requires_grad_(False)
_ = q_target_model2.requires_grad_(False)

alpha = 0.1

opt_pi = torch.optim.AdamW(pi_model.parameters(), lr = 0.0005)

class Categorical:
    def __init__(self, s):
        logits = pi_model(s)
        self._prob = F.softmax(logits, dim = -1)
        self._logp = torch.log(self._prob)

    def prob(self):
        return self._prob
    
    def logp(self):
        return self._logp
    

def optimize_theta(states):
    states = torch.tensor(states, dtype=torch.float64).to(device)

    for p in q_origin_model1.parameters():
        p.requires_grad = False
    
    opt_pi.zero_grad()
    dist = Categorical(states)
    q_value = q_origin_model1(states)
    term1 = dist.prob()
    term2 = q_value - alpha * dist.logp()
    expectation = term1.unsqueeze(1) @ term2.unsqueeze(dim=2)
    expectation = expectation.squeeze(dim=1)
    (-expectation).sum().backward()

    opt_pi.step()

    for p in q_origin_model1.parameters():
        p.requires_grad = True

gamma = 0.99

opt_q1 = torch.optim.AdamW(q_origin_model1.parameters(), lr = 0.0005)
opt_q2 = torch.optim.AdamW(q_origin_model2.parameters(), lr = 0.0005)

def optimize_phi(states, actions, rewards, next_states, dones):
    states = torch.tensor(states, dtype=torch.float64).to(device)
    actions = torch.tensor(actions, dtype=torch.int64).to(device)
    rewards = torch.tensor(rewards, dtype=torch.float64).to(device)
    rewards.unsqueeze(dim=1)
    next_states = torch.tensor(next_states, dtype=torch.float64).to(device)
    dones = torch.tensor(dones, dtype=torch.float64).to(device)
    dones = dones.unsqueeze(dim=1)

    with torch.no_grad():
        q1_tgt_next = q_target_model1(next_states)
        q2_tgt_next = q_target_model2(next_states)
        dist_next = Categorical(next_states)
        q1_target = q1_tgt_next.unsqueeze(dim=1) @ dist_next.prob().unsqueeze(dim=2)
        q1_target = q1_target.squeeze(dim=1)
        q2_target = q2_tgt_next.unsqueeze(dim=1) @ dist_next.prob().unsqueeze(dim=2)
        q2_target = q2_target.squeeze(dim=1)
        q_target_min = torch.minimum(q1_target, q2_target)

        h = dist_next.prob().unsqueeze(dim=1) @ dist_next.logp().unsqueeze(dim=2)
        h = h.squeeze(1)
        h = -alpha * h
        term2 = rewards.unsqueeze(1) + gamma * (1.0 - dones) * (q_target_min + h)

    opt_q1.zero_grad()
    one_hot_actions = F.one_hot(actions, num_classes=2).double()
    q_value1 = q_origin_model1.forward(states)
    term1 = q_value1.unsqueeze(dim=1) @ one_hot_actions.unsqueeze(dim=2)
    term1 = term1.squeeze(dim=1)
    loss_q1 = F.mse_loss(
        term1, 
        term2,
        reduction='none'
    )
    loss_q1.sum().backward()
    opt_q1.step()

    opt_q2.zero_grad()
    one_hot_actions = F.one_hot(actions, num_classes=2).double()
    q_value2 = q_origin_model2.forward(states)
    term1 = q_value2.unsqueeze(dim=1) @ one_hot_actions.unsqueeze(dim=2)
    term1 = term1.squeeze(dim=1)
    loss_q2 = F.mse_loss(
        term1, 
        term2,
        reduction='none'
    )
    loss_q2.sum().backward()
    opt_q2.step()

tau = 0.002

def updater(origin_model:nn.Module, target_model:nn.Module):
    for var, var_target in zip(origin_model.parameters(), target_model.parameters()):
        var_target.data = tau * var.data + (1.0 - tau) * var_target.data

def update_target():
    updater(q_origin_model1, q_target_model1)
    updater(q_origin_model2, q_target_model2)

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
    
    def length(self) -> int:
        return len(self.buffer)
    
buffer = ReplayBuffer(20000)

env = gym.make("CartPole-v1", render_mode='rgb_array')

env = gym.wrappers.RecordVideo(env, video_folder='/home/ago/dai_proj/DAIproject/tmp/video',episode_trigger=lambda x: False)


with torch.no_grad():
    s, _ = env.reset()
    env.start_recording("init")
    done = False
    cum_reward = 0
    while not done:
        a = pick_sample(s)
        s_next, r, term, trunc, _ = env.step(a)
        done = term or trunc
        cum_reward += r
        s = s_next
    env.stop_recording()
    env.reset()
    env.close()


batch_size = 300


env = gym.make("CartPole-v1")
reward_records = []

for i in range(1000):
    s, _ = env.reset()
    done = False
    cum_reward = 0
    while not done:
        a = pick_sample(s)
        s_next, r, term, trunc, _ = env.step(a)
        done = term or trunc
        buffer.add([s.tolist(), a, r, s_next.tolist(), float(term)])
        cum_reward += r
        if buffer.length() >= batch_size:
            states, actions, rewards, n_states, dones = buffer.sample(batch_size)
            optimize_theta(states)
            optimize_phi(states, actions, rewards, n_states, dones)
            update_target()
        s = s_next
    

    reward_records.append(cum_reward)
    print(f"Run episode {i + 1}, with reward {cum_reward}, mean: {np.average(reward_records)}", end='\r')

    if np.average(reward_records[-50:]) > 475.0:
        break

env.close()
print('\nDone')



env = gym.make("CartPole-v1", render_mode='rgb_array')

env = gym.wrappers.RecordVideo(env, video_folder='/home/ago/dai_proj/DAIproject/tmp/video',episode_trigger=lambda x: False)


with torch.no_grad():
    s, _ = env.reset()
    env.start_recording("init")
    done = False
    cum_reward = 0
    while not done:
        a = pick_sample(s)
        s_next, r, term, trunc, _ = env.step(a)
        done = term or trunc
        cum_reward += r
        s = s_next
    env.stop_recording()
    env.reset()
    env.close()