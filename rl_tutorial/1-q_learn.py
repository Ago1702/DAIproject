import gymnasium as gym
import numpy as np

import math

env = gym.make("CartPole-v1")

new_obs_shape = (20, 20, 20, 20)

bins = []

for i in range(4):
    item = np.linspace(
        env.observation_space.low[i] if (i == 0) or (i == 2) else -4,
        env.observation_space.high[i] if (i == 0) or (i == 2) else 4,
        num=new_obs_shape[i],
        endpoint=False
    )
    item = np.delete(item, 0)
    bins.append(item)

    print(bins[i])


def get_discrete_state(s:list):
    new_s = []
    for i in range(4):
        new_s.append(np.digitize(s[i], bins[i]))
    
    return new_s

q_table = np.zeros(new_obs_shape + (env.action_space.n,))
print(q_table.shape)


gamma = 0.99
alpha = 0.1
eps = 1
eps_decay = eps / 4000

def pickup_sample(s, eps:float):
    if np.random.random() > eps:
        a = np.argmax(q_table[tuple(s)])
    else:
        a = np.random.randint(0, env.action_space.n)
    return a


env = gym.make("CartPole-v1")
reward_records = []

for i in range(6000):

    done = False
    total_reward = 0

    s, _ = env.reset()

    s_dis = get_discrete_state(s)
    while not done:
        a = pickup_sample(s_dis, eps)
        s, r, term, trunc, _ = env.step(a)
        done = term or trunc
        s_dis_next = get_discrete_state(s)

        maxQ = np.max(q_table[tuple(s_dis_next)])
        q_table[tuple(s_dis)][a] += alpha * (r + gamma * maxQ - q_table[tuple(s_dis)][a])

        s_dis = s_dis_next

        total_reward += r
    
    if eps - eps_decay >= 0:
        eps -= eps_decay
    
    if (i + 1) % 1000 == 0:
        print(f"Run episode {i + 1} with rewards {total_reward}")
    
    reward_records.append(total_reward)
    
print("\nDone")

env = env = gym.make("CartPole-v1", render_mode='human', max_episode_steps=1000)
s, _ = env.reset()

done = False
s_dis = get_discrete_state(s)
while not done:
    a = pickup_sample(s_dis, eps)
    s, r, term, trunc, _ = env.step(a)
    done = term or trunc
    s_dis_next = get_discrete_state(s)

    maxQ = np.max(q_table[tuple(s_dis_next)])
    q_table[tuple(s_dis)][a] += alpha * (r + gamma * maxQ - q_table[tuple(s_dis)][a])

    s_dis = s_dis_next

    total_reward += r


env.close()

import matplotlib.pyplot as plt
# Generate recent 50 interval average
average_reward = []
for idx in range(len(reward_records)):
    avg_list = np.empty(shape=(1,), dtype=int)
    if idx < 50:
        avg_list = reward_records[:idx+1]
    else:
        avg_list = reward_records[idx-49:idx+1]
    average_reward.append(np.average(avg_list))
# Plot
plt.plot(reward_records)
plt.plot(average_reward)

plt.savefig("./tmp/qlearn.png", format='png')