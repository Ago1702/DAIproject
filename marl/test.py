PRINT = False

import copy
import tempfile

import torch

from matplotlib import pyplot as plt
from tensordict import TensorDictBase

from tensordict.nn import TensorDictModule, TensorDictSequential
from torch import multiprocessing

from torchrl.collectors import SyncDataCollector
from torchrl.data import LazyMemmapStorage, RandomSampler, ReplayBuffer

from torchrl.envs import (
    check_env_specs,
    ExplorationType,
    PettingZooEnv,
    RewardSum,
    set_exploration_type,
    TransformedEnv,
    VmasEnv,
)

from torchrl.modules import (
    AdditiveGaussianModule,
    MultiAgentMLP,
    ProbabilisticActor,
    TanhDelta,
)

from torchrl.objectives import DDPGLoss, SoftUpdate, ValueEstimators
from torchrl.record import CSVLogger, PixelRenderTransform, VideoRecorder

from tqdm import tqdm

try:
    is_sphinx = __sphinx_build__
except NameError:
    is_sphinx = False

#########CODE########

seed = 2025
torch.manual_seed(seed)


is_fork = multiprocessing.get_start_method() == 'fork'
device = (
    torch.device(0)
    if torch.cuda.is_available() and not is_fork
    else torch.device('cpu')
)
print('',"+++" + str(device) + "+++",'')

frames_per_batch = 1_000
n_iters = 10
total_frames = frames_per_batch * n_iters

iteration_when_stop_training_evaders = n_iters // 2

memory_size = 1_000_000

n_optimizer_step = 100
tain_batch_size = 128
lr = 3e-4
max_grad_norm = 1.0

gamma = 0.99
polyak_tau = 0.005

### ENV ###

max_steps = 100

n_chasers = 2
n_evaders = 1
n_obstacles = 2

use_vmas = False

if not use_vmas:
    kwargs = {  #Scenario Spec
        "continuous_actions":True,
        "num_good":n_evaders,
        "num_adversaries":n_chasers,
        "num_obstacles":n_obstacles,
        "max_cycles":max_steps,
    }
    base_env = PettingZooEnv(
        task='simple_tag_v3',
        parallel=True,
        seed=seed,
        **kwargs
    )
else:
    num_vmas_envs = (
        frames_per_batch // max_steps
    )
    kwargs = {  #Scenario Spec
        "num_good_agents":n_evaders,
        "num_adversaries":n_chasers,
        "num_landmarks":n_obstacles,
    }
    base_env = VmasEnv(
        scenario="simple_tag",
        num_envs=num_vmas_envs,
        continuous_actions=True,
        max_steps=max_steps,
        device=device,
        seed=seed,
        **kwargs
    )

if PRINT:
    print(f"group_map: {base_env.group_map}",'')
    print("action_spec:", base_env.full_action_spec,'')
    print("reward_spec:", base_env.full_reward_spec,'')
    print("done_spec:", base_env.full_done_spec,'')
    print("observation_spec:", base_env.observation_spec,'')
    print("action_keys:", base_env.action_keys)
    print("reward_keys:", base_env.reward_keys)
    print("done_keys:", base_env.done_keys)

env = TransformedEnv(
    base_env,
    RewardSum(
        in_keys=base_env.reward_keys,
        reset_keys=["_reset"]*len(base_env.group_map.keys()),
    ),
)

check_env_specs(env)

n_rollout_steps = 5
rollout = env.rollout(n_rollout_steps)
if PRINT:
    print(f"rollout of {n_rollout_steps} steps:", rollout)
    print("Shape of the rollout TensorDict:", rollout.batch_size)
