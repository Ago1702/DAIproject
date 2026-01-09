import warnings

warnings.filterwarnings("ignore")

from pettingzoo.mpe import simple_spread_v3
import torch.multiprocessing.spawn
from models import TeamNet, CriticNet

import numpy as np

import torch
from torch import nn as nn
from torch.nn import functional as F

from torchrl.data import ReplayBuffer, ListStorage, LazyTensorStorage, LazyMemmapStorage 

from torch.multiprocessing import Process

from copy import deepcopy

from env_wrp import EnvWrapper

import evolution as evo

import wandb

import random

import os
import sys



def update_target_params(network:nn.Module, target_network:nn.Module, tau:float = 0.1) -> None:
    for var, target_var in zip(network.parameters(), target_network.parameters()):
        target_var.data = tau * var.data + (1.0 - tau) * target_var.data



POP_SIZE = 10
ROLL_SIZE = 5
TAU = 0.01
ACTOR_LR = 0.01
CRITIC_LR = 0.01
GAMMA = 0.95
REPLAY_SIZE = 1e6
BATCH = 2048

GRADIENT_SIZE = 0

CROSS_PROB = 0.4
MUT_PROB = 0.9
MUT_FRAC = 0.1
MUT_STR = 0.1
S_MUT_PROB = 0.05
RESET_MUT_PROB = 0.05

ELITES_NUM = 4

FIT_ROLL = 5

WANDB = False

PATH = '/home/ago/dai_proj/DAIproject/project/save'

class BufferCont:
    def __init__(self, buffers):
        self.episodes = [buffer for buffer in buffers]


def _filler_(team, device, buffers, lock):
    print('Suca')
    return 0
    with torch.no_grad():
        fit = 0
        env1 = EnvWrapper()
        team.cpu()
        for j in range(FIT_ROLL):
            reward_history = env1.fill_buffer(buffers.episodes, team, device=device, lock=lock)
            fit += np.sum(reward_history)
        fit /= FIT_ROLL
        env1.fill_buffer(buffers, team, device=device, noise=True)
        del env1
        team.to(device)
        print('Suca fine')
        return 0
        #return fit

if __name__ == '__main__':
    warnings.filterwarnings("ignore")

    #Params retrival
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    env = EnvWrapper()

    if WANDB:
        run = wandb.init(
            entity='davide-agostini1702-university-of-modena-and-reggio-emilia',
            project='EVO_A-test',
        )
    for agent in env.agents:
        input_size = env.env.observation_space(agent)._shape[0]
        output_size = env.env.action_space(agent).shape[0]
        output_min = env.env.action_space(agent).low.tolist()
        output_min = torch.tensor(output_min)
        output_max = env.env.action_space(agent).high.tolist()
        output_max = torch.tensor(output_max)


    #Initialize nets
    evo_teams:list[TeamNet] = []

    for i in range(POP_SIZE):
        evo_teams.append(TeamNet(len(env.agents), input_size, output_size).to(device))

    gradient_teams:list[TeamNet] = []
    target_teams:list[TeamNet] = []
    actors_optim = []
    for i in range(GRADIENT_SIZE):
        team = TeamNet(len(env.agents), input_size, output_size).to(device)
        target_team = TeamNet(len(env.agents), input_size, output_size).to(device)
        update_target_params(team, target_team, 1.0)
        actors_optim.append(torch.optim.Adam(team.parameters(), ACTOR_LR))
        gradient_teams.append(team)
        target_teams.append(target_team)

    critic = CriticNet(input_size, output_size).to(device)
    target_critic = CriticNet(input_size, output_size).to(device)

    update_target_params(critic, target_critic, 1.0)

    critic_optim = torch.optim.Adam(critic.parameters(), CRITIC_LR)


    buffers = []
    for id in range(len(env.agents)):
        buffers.append(ReplayBuffer(storage=LazyMemmapStorage(REPLAY_SIZE), shared=True))

    q_step = 0

    gen = 0

    while True:
        fitness = [0.] * POP_SIZE
        with torch.no_grad():
            for i, pop in enumerate(evo_teams):
                evo_teams[i].cpu()
                for j in range(FIT_ROLL):
                    reward_history = env.fill_buffer(buffers, evo_teams[i], device=device)
                    fitness[i] += np.sum(reward_history)
                #print(f"fitness {pop} is {fitness[pop]}")
                fitness[i] /= FIT_ROLL
                for j in range(FIT_ROLL//5):
                    reward_history = env.fill_buffer(buffers, evo_teams[i], device=device, noise=True)
                evo_teams[i].to(device)
        print(len(buffers[0]))
        '''func_input = [[team, device, buffers] for team in evo_teams]
        with ThreadPool() as pool:
            fitness = pool.map(filler, func_input)'''
        processes:list[Process] = []
        for team in evo_teams:
            args = {'team':team, "device":device, "buffers":buffers[0], "lock":True}
            p = Process(target=_filler_, kwargs=args)
            p.start()
            processes.append(p)
            break
        for p in processes:
            p.join()

        
        print(len(buffers[0]))
        sys.exit()

        print(f"Gen {gen} mean reward is: {np.mean(fitness).item():.5f}, len: {len(fitness)} pop: {POP_SIZE}")
        if WANDB:
            run.log({'avg_reward':np.mean(fitness).item()}, step=gen)
        ranking = np.argsort(fitness).tolist()
        ranking.reverse()
        elites = ranking[:ELITES_NUM]

        offsprings = evo.tournament_selection(ranking, len(ranking) - len(elites) - GRADIENT_SIZE, tournament_size=3)


        unselcted = []
        new_elites = []

        for i in range(POP_SIZE):
            if i in offsprings or i in elites:
                continue
            else:
                unselcted.append(i)

        random.shuffle(unselcted)

        for team in target_teams:
            if len(unselcted) >= 1:
                replacee = unselcted.pop(0)
                update_target_params(team, evo_teams[replacee], 1.0)

        for i in elites:
            if len(unselcted) >= 1: replacee = unselcted.pop(0)
            elif len(offsprings) >= 1: replacee = offsprings.pop(0)
            else: continue

            new_elites.append(replacee)
            update_target_params(evo_teams[i], evo_teams[replacee], 1.0)

        if len(unselcted) % 2 != 0:
            unselcted.append(unselcted[random.randint(0, len(unselcted) - 1)])

        for i, j in zip(unselcted[0::2], unselcted[1::2]):
            off_i = random.choice(new_elites)
            off_j = random.choice(offsprings)

            update_target_params(evo_teams[off_i], evo_teams[i], 1.0)
            update_target_params(evo_teams[off_j], evo_teams[j], 1.0)
            evo.crossover_inplace(evo_teams[i], evo_teams[j])

        for i, j in zip(offsprings[0::2], offsprings[1::2]):
            if random.random() < CROSS_PROB:
                evo.crossover_inplace(evo_teams[i], evo_teams[j])

        for i in range(POP_SIZE):
            if i not in new_elites:
                if random.random() < MUT_PROB:
                    evo.mutate_inplace(evo_teams[i])

        # Evo Algo end
        c_loss_history = []
        for i in range(GRADIENT_SIZE):
            q_loss = 0
            actor_loss_history = []
            for agent in range(len(env.agents)):
                obs, act, rew, new_obs = buffers[agent].sample(BATCH)
                with torch.no_grad():
                    new_act = target_teams[i].noise_action(new_obs, agent)
                    Q1, Q2 = target_critic.forward(new_obs, new_act)
                    val = torch.min(Q1, Q2, out=None).squeeze(1)
                    q_val = rew + GAMMA * val
                Q1_, Q2_ = critic.forward(obs, act)
                q_res = torch.min(Q1_, Q2_).squeeze(1)
                q_loss = q_loss + F.mse_loss(q_res, q_val)

                actor_action = gradient_teams[i].action(obs, agent)
                Q1, Q2 = critic.forward(obs, actor_action)
                act_loss = -torch.min(Q1, Q2)
                act_loss = torch.mean(act_loss)
                actors_optim[i].zero_grad()
                act_loss.backward(retain_graph=True)
                actors_optim[i].step()
                actor_loss_history.append(act_loss.item())

            if WANDB:
                run.log({f'avg_team{i}_loss':np.mean(actor_loss_history).item()}, step=gen)
            else:
                loss_mean = np.mean(actor_loss_history).item()

            q_loss = q_loss / 3
            critic_optim.zero_grad()
            q_loss.backward(retain_graph=True)
            critic_optim.step()
            update_target_params(gradient_teams[i], target_teams[i], 0.05)
            c_loss_history.append(q_loss.item())

        if WANDB:
                run.log({f'critic_loss':np.mean(c_loss_history).item()}, step=gen)
        update_target_params(critic, target_critic, 0.01)    
        #END LEARNING

        gen += 1

        if (gen) % 5000 == 0:
            torch.save(critic.state_dict(), f=f"{PATH}/critic")
            torch.save(target_critic.state_dict(), f=f"{PATH}/target-critic")
            for i in range(GRADIENT_SIZE):
                torch.save(gradient_teams[i].state_dict(), f=f"{PATH}/gr-team{i}")
                torch.save(target_teams[i].state_dict(), f=f"{PATH}/gr-target{i}")
            for i in range(POP_SIZE):
                torch.save(evo_teams[i].state_dict(), f=f"{PATH}/evo-model{i}-el{elites[0]}")