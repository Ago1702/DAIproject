import random, sys
import numpy as np
import math
import torch
import torch.nn as nn

MUT_PROB = 0.9
MUT_FRAC = 0.1
MUT_STR = 0.1
S_MUT_STR = 10
S_MUT_PROB = 0.05
RESET_MUT_PROB = 0.07
W_CLAMP = 10000


def tournament_selection(index_rank, num_offsprings, tournament_size):
    
    total_choices = len(index_rank)
    offsprings = []

    for i in range(num_offsprings):
        winner = np.min(np.random.randint(total_choices, size=tournament_size))
        offsprings.append(index_rank[winner])
    
    offsprings = list(set(offsprings))

    if len(offsprings) % 2 != 0:
        offsprings.append(offsprings[np.random.randint(0, len(offsprings))])
    
    return offsprings

def reg_weight(weight, mag):
    if weight > mag: weight = mag
    if weight < -mag: weight = -mag

    return weight

def crossover_inplace(gene1:nn.Module, gene2:nn.Module):

    keys1 = list(gene1.state_dict())
    keys2 = list(gene2.state_dict())

    for key in keys1:
        if key not in keys2:
            continue

        W1 = gene1.state_dict()[key]
        W2 = gene2.state_dict()[key]

        if len(W1.shape) == 2:
            num_variables = W1.shape[0]

            try: num_cross = random.randint(0, int(num_variables * 0.3))
            except: num_cross = 1

            for i in range(num_cross):
                receiver_choice = random.random()

                if receiver_choice < 0.5:
                    ind_cr = random.randint(0, num_variables - 1)
                    W1[ind_cr, :] = W2[ind_cr, :]
                else:
                    ind_cr = random.randint(0, num_variables - 1)
                    W2[ind_cr, :] = W1[ind_cr, :]
        
        elif len(W1.shape) == 1:
            if random.random() < 0.8: continue
            num_variables = W1.shape[0]
            
            num_cross = 1

            for i in range(num_cross):
                receiver_choice = random.random()

                if receiver_choice < 0.5:
                    ind_cr = random.randint(0, num_variables - 1)
                    W1[ind_cr] = W2[ind_cr]
                else:
                    ind_cr = random.randint(0, num_variables - 1)
                    W2[ind_cr] = W1[ind_cr]


def mutate_inplace(gene:nn.Module):
    num_params = len(list(gene.parameters()))
    ssne_probs = np.random.uniform(0, 1, num_params) * 2

    for i, param in enumerate(gene.parameters()):

        W = param.data
        if len(W.shape) == 2:
            num_w = W.shape[0] * W.shape[1]

            ssne_prob = ssne_probs[i]

            if random.random() < ssne_prob:
                num_mutations = random.randint(0, int(math.ceil(MUT_FRAC * num_w)))

                for _ in range(num_mutations):
                    ind_dim1 = random.randint(0, W.shape[0] - 1)
                    ind_dim2 = random.randint(0, W.shape[1] - 1)

                    random_num = random.random()

                    if random_num < S_MUT_PROB:
                        W[ind_dim1, ind_dim2] += random.gauss(0, S_MUT_STR * W[ind_dim1, ind_dim2])
                    elif random_num < RESET_MUT_PROB:
                        W[ind_dim1, ind_dim2] = random.gauss(0, 0.1)
                    else:
                        W[ind_dim1, ind_dim2] += random.gauss(0, MUT_STR * W[ind_dim1, ind_dim2])
                    
                    W[ind_dim1, ind_dim2] = reg_weight(W[ind_dim1, ind_dim2], W_CLAMP)
                
        elif len(W.shape) == 1:
            num_w = W.shape[0]

            ssne_prob = ssne_probs[i] * 0.04

            if random.random() < ssne_prob:
                num_mutations = random.randint(0, int(math.ceil(MUT_FRAC * num_w)))

                for _ in range(num_mutations):
                    ind_dim = random.randint(0, W.shape[0] - 1)

                    random_num = random.random()

                    if random_num < S_MUT_PROB:
                        W[ind_dim] += random.gauss(0, S_MUT_STR * W[ind_dim])
                    elif random_num < RESET_MUT_PROB:
                        W[ind_dim] = random.gauss(0, 1)
                    else:
                        W[ind_dim] += random.gauss(0, MUT_STR * W[ind_dim])
                    
                    W[ind_dim] = reg_weight(W[ind_dim], W_CLAMP)
        

def reset_genome(self, gene:nn.Module):
    for param in gene.parameters():
        param.data.copy_(param.data)

def roulette_wheel(probs, num_samples):

    probs = [prob - min(probs) + abs(min(probs)) for prob in probs]

    total_prob = sum(probs) if sum(probs) != 0 else 1.0

    probs = [prob/total_prob for prob in probs]

    out = []

    for _ in range(num_samples):
        rand = random.random()

        for i in range(len(probs)):
            if rand < sum(probs[0:i + 1]):
                out.append(i)
                break
    
    return out