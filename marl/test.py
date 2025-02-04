import torch

from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor

from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage

from torchrl.envs import RewardSum, TransformedEnv
from torchrl.envs.libs.pettingzoo import PettingZooEnv
from torchrl.envs.libs.vmas import VmasEnv
from torchrl.envs.utils import check_env_specs

from torchrl.modules import MultiAgentMLP, ProbabilisticActor, TanhNormal

from torchrl.objectives import ClipPPOLoss, ValueEstimators

torch.manual_seed(2025)
#from matplotlib import pyplot as plt
#from tqdm import tqdm

device = "cpu" if not torch.backends.cuda.is_built() else "cuda:0"
print(device)

frames_per_batch = 6_000
n_iters = 10
total_frames = frames_per_batch * n_iters

num_epochs = 30
minibatch_size = 400
lr = 3e-4
max_grad_norm = 1.0

clip_epsilon = 0.2
gamma = 0.9
lmbda = 0.9
entropy_eps = 1e-4

max_steps = 100
num_ptzoo_envs = (
    frames_per_batch // max_steps
)

scenario_name = 'simple_adversary_v3'
n_agents = 3
kwargs = {"n_pistons": 21, "continuous": True}

#env = VmasEnv(
#    scenario=scenario_name,
#    num_envs=num_vmas_envs,
#    continuous_actions=True,  # VMAS supports both continuous and discrete actions
#    max_steps=max_steps,
#    device=vmas_device,
#    # Scenario kwargs
#    n_agents=n_agents,  # These are custom kwargs that change for each VMAS scenario, see the VMAS repo to know more.
#)

env = PettingZooEnv(
    task=scenario_name,
    parallel=True,
    return_state=True,
    group_map=None,
    seed=2025
)
print(env.reward_keys)
env = TransformedEnv(  # RewardSum transform which will sum rewards over the episode.
    env,
    RewardSum(in_keys=env.reward_keys, out_keys=[("adversary", "episode_reward"), ("agents", "episode_reward")]),
)
check_env_specs(env)