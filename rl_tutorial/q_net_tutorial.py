import random
import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as f

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class QNet(nn.Module):

    def __init__(self, hidden_dim:int = 64):
        super(QNet, self).__init__()

        self.hidden = nn.Linear(4, hidden_dim)
        self.output = nn.Linear(hidden_dim, 2)
    
    def forward(self, s:torch.Tensor):
        outs = self.hidden(s)
        outs = f.relu(outs)
        outs = self.output(outs)
        return outs
    
q_model = QNet().to(device)
q_target_model = QNet().to(device)

q_target_model.load_state_dict(q_model.state_dict())

_ = q_target_model.requires_grad_(False)

class ReplayMemory:
    def __init__(self, buffer_size:int):
        self.buffer_size = buffer_size
        self.buffer = []

    def add(self, item):
        if len(self.buffer) == self.buffer_size:
            self.buffer.pop(0)
        self.buffer.append(item)
    
    def sample(self, sample_size:int):

        items = random.sample(self.buffer, sample_size)

        states   = [i[0] for i in items]
        actions  = [i[1] for i in items]
        rewards  = [i[2] for i in items]
        n_states = [i[3] for i in items]
        dones    = [i[4] for i in items]

        states = torch.tensor(states, dtype=torch.float).to(device)
        actions = torch.tensor(actions, dtype=torch.int64).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float).to(device)
        n_states = torch.tensor(n_states, dtype=torch.float).to(device)
        dones = torch.tensor(dones, dtype=torch.float).to(device)

        return states, actions, rewards, n_states, dones
    
    def __len__(self):
        return len(self.buffer)


memory = ReplayMemory(buffer_size=10000)

gamma = 0.99

opt = torch.optim.Adam(q_model.parameters(), lr=0.0005)

def optimize(states:torch.Tensor, actions:torch.Tensor, rewards:torch.Tensor, next_states:torch.Tensor, dones:torch.Tensor):
    
    with torch.no_grad():
        target_vals_for_all_actions = q_target_model(next_states)

        target_actions = torch.argmax(target_vals_for_all_actions, 1)

        target_actions_one_hot = f.one_hot(target_actions, env.action_space.n).float()
        target_vals = torch.sum(target_vals_for_all_actions * target_actions_one_hot, 1)

        target_vals_masked = (1.0 - dones) * target_vals

        q_vals1 = rewards + gamma * target_vals_masked

    opt.zero_grad()

    actions_one_hot = f.one_hot(actions, env.action_space.n).float()
    q_vals2 = torch.sum(q_model(states) * actions_one_hot, 1)

    loss = f.mse_loss(
        q_vals1.detach(),
        q_vals2,
        reduction='mean'
    )

    loss.backward()

    opt.step()

sampling_size = 64 * 30
batch_size = 64

eps = 1.0
eps_decay = eps / 3000
eps_final = 0.1

env = gym.make("CartPole-v1")

def pickup_sample(s, eps):
    with torch.no_grad():

        if np.random.random() > eps:
            s_batch = torch.tensor(s, dtype=torch.float).to(device)
            s_batch = s_batch.unsqueeze(0)
            q_vals_all_actions = q_model(s_batch)
            a = torch.argmax(q_vals_all_actions, 1)
            a = a.squeeze(0)
            a = a.tolist()
        else:
            a = np.random.randint(0, env.action_space.n)

        return a

def evaluate():
    with torch.no_grad():
        s, _ = env.reset()

        done = False
        total = 0
        while not done:
            a = pickup_sample(s, 0.0)
            s_next, r, term, trunc, _ = env.step(a)
            done = term or trunc
            total += r
            s = s_next
        return total
    
reward_records = []

mean = 0

for i in range(15000):
    done = True
    for _ in range(500):
        if done:
            s, _ = env.reset()
            done = False
            cum_reward = 0
        
        a = pickup_sample(s, eps)
        s_next, r, term, trunc, _ = env.step(a)
        done = term or trunc
        memory.add([s.tolist(), a, r, s_next.tolist(), float(term)])
        cum_reward += r
        s = s_next
    
    if memory.__len__() < 2000:
        continue

    states, actions, rewards, n_states, dones = memory.sample(sampling_size)

    states = torch.reshape(states, (-1, batch_size, 4))
    actions = torch.reshape(actions, (-1, batch_size))
    rewards = torch.reshape(rewards, (-1, batch_size))
    n_states = torch.reshape(n_states, (-1, batch_size, 4))
    dones = torch.reshape(dones, (-1, batch_size))

    for j in range(actions.size(0)):
        optimize(states[j], actions[j], rewards[j], n_states[j], dones[j])
    
    total_reward = evaluate()
    reward_records.append(total_reward)
    iteration_num = len(reward_records)
    if (iteration_num + 1) % 250 == 0:
        mean = np.average(reward_records)

    print(f"Iteration {iteration_num + 1}, Reward {total_reward}, eps {eps:1.3f}, mean = {mean}", end='\r')

    if iteration_num % 50 == 0:
        q_target_model.load_state_dict(q_model.state_dict())
    
    if eps - eps_decay >= eps_final:
        eps -= eps_decay
    
    if np.average(reward_records[-200:]) >= 495.0:
        break

env.close()
print('\nDone')

#una bella prova

env = gym.make("CartPole-v1", render_mode='human')

done = False

s, _ = env.reset()

while not done:
    a = pickup_sample(s, 0.0)
    s_next, r, term, trunc, _ = env.step(a)
    done = term or trunc
    memory.add([s.tolist(), a, r, s_next.tolist(), float(term)])
    cum_reward += r
    s = s_next