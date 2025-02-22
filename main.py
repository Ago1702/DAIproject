from pettingzoo.butterfly import cooperative_pong_v5
import torch

device = "cpu" if not torch.backends.cuda.is_built() else "cuda:0"
print(device)
env = cooperative_pong_v5.env(render_mode="human")
env.reset(seed=42)

for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()

    if termination or truncation:
        action = None
    else:
        # this is where you would insert your policy
        action = env.action_space(agent).sample()

    env.step(action)
env.close()