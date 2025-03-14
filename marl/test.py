PRINT=True

import os
import copy
import tempfile

import torch

from matplotlib import pyplot as plt
from tensordict import TensorDictBase

from tensordict.nn import TensorDictModule, TensorDictSequential
from torch import multiprocessing

from torchrl.collectors import SyncDataCollector
from torchrl.data import LazyTensorStorage, RandomSampler, ReplayBuffer

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


is_fork = multiprocessing.get_start_method() == "fork"
device = (
    torch.device(0)
    if torch.cuda.is_available() and not is_fork
    else torch.device("cpu")
)

frames_per_batch = 48_000
n_iters = 100
total_frames = frames_per_batch * n_iters

iteration_when_stop_training_evaders = n_iters // 3 * 2

memory_size = 1_000_000

n_optimiser_steps = 100
train_batch_size = 1024
lr = 3e-4  # Learning rate
max_grad_norm = 1.0

gamma = 0.99
polyak_tau = 0.005

### ENV ###

max_steps = 100  # Environment steps before done

n_chasers = 2
n_evaders = 1
n_obstacles = 2

use_vmas = True

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
        **kwargs,
        device=device,
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
        **kwargs,
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

#NN definition

policy_modules = {}
for group, agents in env.group_map.items():
    share_parameters_policy = True
    policy_net = MultiAgentMLP(
        n_agent_inputs=env.observation_spec[group, "observation"].shape[
            -1
        ],
        n_agent_outputs=env.full_action_spec[group, "action"].shape[-1],
        n_agents=len(agents),
        centralised=False,
        share_params=share_parameters_policy,
        device=device,
        depth=2,
        num_cells=256,
        activation_class=torch.nn.Tanh,
    )

    policy_module = TensorDictModule(
        policy_net,
        in_keys=[(group, "observation")],
        out_keys=[(group, "param")],
    )

    policy_modules[group] = policy_module

policies = {}

for group, _agents in env.group_map.items():
    policy = ProbabilisticActor(
        module=policy_modules[group],
        spec=env.full_action_spec[group, "action"],
        in_keys=[(group, "param")],
        out_keys=[(group, "action")],
        distribution_class=TanhDelta,
        distribution_kwargs={
            "low": env.full_action_spec[group, "action"].space.low,
            "high": env.full_action_spec[group, "action"].space.high,
        },
        return_log_prob=False,
    )
    policies[group] = policy

exploration_policies = {}
for group, _agents in env.group_map.items():
    exploration_policy = TensorDictSequential(
        policies[group],
        AdditiveGaussianModule(
            spec=policies[group].spec,
            annealing_num_steps=total_frames//2,
            action_key=(group, 'action'),
            sigma_init=0.9,
            sigma_end=0.1,
        ),
    )
    exploration_policies[group] = exploration_policy

##Critic Network

critics = {}

for group, agents in env.group_map.items():
    share_parameters_critic = True
    MADDPG = True

    cat_module = TensorDictModule(
        lambda obs, action: torch.cat([obs, action], dim=-1),
        in_keys=[(group, "observation"), (group, "action")],
        out_keys=[(group, "obs_action")],
    )

    critic_module = TensorDictModule(
        module=MultiAgentMLP(
            n_agent_inputs=env.observation_spec[group, "observation"].shape[-1]
            + env.full_action_spec[group, "action"].shape[-1],
            n_agent_outputs=1,  # 1 value per agent
            n_agents=len(agents),
            centralised=MADDPG,
            share_params=share_parameters_critic,
            device=device,
            depth=2,
            num_cells=256,
            activation_class=torch.nn.Tanh,
        ),
        in_keys=[(group, "obs_action")],
        out_keys=[
            (group, "state_action_value")
        ],  # Write ``(group, "state_action_value")``
    )

    critics[group] = TensorDictSequential(
        cat_module, critic_module
    )  # Run them in sequence

reset_td = env.reset()
for group, _agents in env.group_map.items():
    print(
        f"Running value and policy for group '{group}':",
        critics[group](policies[group](reset_td)),
    )
# Put exploration policies from each group in a sequence
agents_exploration_policy = TensorDictSequential(*exploration_policies.values())

collector = SyncDataCollector(
    env,
    agents_exploration_policy,
    device=device,
    frames_per_batch=frames_per_batch,
    total_frames=total_frames,
)

replay_buffers = {}
for group, _agents in env.group_map.items():
    replay_buffer = ReplayBuffer(
        storage=LazyTensorStorage(
            memory_size, device=device
        ),
        sampler=RandomSampler(),
        batch_size=train_batch_size,
    )
    replay_buffers[group] = replay_buffer

losses = {}

for group, _agents in env.group_map.items():
    loss_module = DDPGLoss(
        actor_network=policies[group],
        value_network=critics[group],
        delay_value=True,
        loss_function="l2",

    )
    loss_module.set_keys(
        state_action_value=(group, "state_action_value"),
        reward=(group, "reward"),
        done=(group, "done"),
        terminated=(group, "terminated"),
    )
    loss_module.make_value_estimator(ValueEstimators.TD0, gamma=gamma)

    losses[group] = loss_module

target_updaters = {
    group: SoftUpdate(loss, tau=polyak_tau) for group, loss in losses.items()
}

optimisers = {
    group: {
        "loss_actor": torch.optim.Adam(
            loss.actor_network_params.flatten_keys().values(), lr=lr
        ),
        "loss_value": torch.optim.Adam(
            loss.value_network_params.flatten_keys().values(), lr=lr
        ),
    }
    for group, loss in losses.items()
}

def process_batch(batch: TensorDictBase) -> TensorDictBase:
    for group in env.group_map.keys():
        keys = list(batch.keys(True, True))
        group_shape = batch.get_item_shape(group)
        nested_done_key = ("next", group, "done")
        nested_terminated_key = ("next", group, "terminated")
        if nested_done_key not in keys:
            batch.set(
                nested_done_key,
                batch.get(("next", "done")).unsqueeze(-1).expand((*group_shape, 1)),
            )
        if nested_terminated_key not in keys:
            batch.set(
                nested_terminated_key,
                batch.get(("next", "terminated"))
                .unsqueeze(-1)
                .expand((*group_shape, 1)),
            )
    return batch

pbar = tqdm(
    total=n_iters,
    desc=", ".join(
        [f"episode_reward_mean_{group} = 0" for group in env.group_map.keys()]
    ),
)
episode_reward_mean_map = {group: [] for group in env.group_map.keys()}
train_group_map = copy.deepcopy(env.group_map)

for iteration, batch in enumerate(collector):
    current_frames = batch.numel()
    batch = process_batch(batch)
    for group in train_group_map.keys():
        group_batch = batch.exclude(
            *[
                key for _group in env.group_map.keys()
                if _group != group
                for key in [_group, ('next', _group)]
            ]
        )
        group_batch = group_batch.reshape(-1)
        replay_buffers[group].extend(group_batch)

        for _ in range(n_optimiser_steps):
            subdata = replay_buffers[group].sample()
            loss_vals = losses[group](subdata)

            for loss_name in ["loss_actor", "loss_value"]:
                loss = loss_vals[loss_name]
                optimiser = optimisers[group][loss_name]

                loss.backward()

                params = optimiser.param_groups[0]["params"]
                torch.nn.utils.clip_grad_norm_(params, max_grad_norm)

                optimiser.step()
                optimiser.zero_grad()
            
            target_updaters[group].step()
        
        exploration_policies[group][-1].step(current_frames)

    if iteration == iteration_when_stop_training_evaders:
        del train_group_map['agent']
    
    for group in env.group_map.keys():
        episode_reward_mean = (
            batch.get(('next', group, "episode_reward"))[
                batch.get(('next', group, 'done'))
            ].mean().item()
        )
        episode_reward_mean_map[group].append(episode_reward_mean)
    
    pbar.set_description(
        ", ".join(
            [
                f"episode_reward_mean_{group} = {episode_reward_mean_map[group][-1]:.4}"
                for group in env.group_map.keys()
            ]
        ),
        refresh=False
    )
    pbar.update()

fig, axs = plt.subplots(2, 1)
for i, group in enumerate(env.group_map.keys()):
    axs[i].plot(episode_reward_mean_map[group], label=f"Episode reward mean {group}")
    axs[i].set_ylabel("Reward")
    axs[i].axvline(
        x=iteration_when_stop_training_evaders,
        label="Agent (evader) stop training",
        color="orange",
    )
    axs[i].legend()
axs[-1].set_xlabel("Training iterations")
plt.savefig("plot.png")

#with torch.no_grad():
#    env.rollout(
#        max_steps=max_steps,
#        policy=agents_exploration_policy,
#        callback=lambda env, _: env.render(),
#        auto_cast_to_device=True,
#        break_when_any_done=True,
#    )

if use_vmas and not is_sphinx:
    # Replace tmpdir with any desired path where the video should be saved
    tmpdir = "./tmp/video"
    video_logger = CSVLogger("vmas_logs", tmpdir, video_format="mp4")
    print("Creating rendering env")
    env_with_render = TransformedEnv(env.base_env, env.transform.clone())
    env_with_render = env_with_render.append_transform(
        PixelRenderTransform(
            out_keys=["pixels"],
            # the np.ndarray has a negative stride and needs to be copied before being cast to a tensor
            preproc=lambda x: x.copy(),
            as_non_tensor=True,
            # asking for array rather than on-screen rendering
            mode="rgb_array",
        )
    )
    env_with_render = env_with_render.append_transform(
        VideoRecorder(logger=video_logger, tag="vmas_rendered", skip=0)
    )
    with set_exploration_type(ExplorationType.DETERMINISTIC):
        print("Rendering rollout...")
        env_with_render.rollout(100, policy=agents_exploration_policy, break_when_any_done=True)
    print("Saving the video...")
    env_with_render.transform.dump()
    print("Saved! Saved directory tree:")
    video_logger.print_log_dir()