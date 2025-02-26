import random, sys
import numpy as np
import math
import torch
from torch import nn
from marl.code.utils import hard_update, soft_update, list_mean

class SSNE:
    
    def __init__(self, args):
        self.args = args
        self.gen = 0

        self.env = args.config.env_choice
        self.pop_size = args.pop_size
        self.cross_prob = args.cross_prob
        self.mut_prob = args.mut_prob
        self.ext_prob = args.ext_prob
        self.ext_mag = args.ext_mag

        self.weight_clamp = args.weight_clamp
        self.mut_dst = args.mut_dst
        self.lin_depth = 10
        self.ccea_red = 'leniency'
        self.num_elites = args.num_elites

        self.lineage = [[] for _ in range(self.pop_size)]
        self.all_offs = []

    def tournament(self, index_rank:list, num_offspring:int, t_size:int):
        
        total_choice = len(index_rank)
        offsprings = []

        for i in range(num_offspring):
            winner = np.min(np.random.randint(total_choice, size=t_size))
            offsprings.append(index_rank[winner])
        
        offsprings = list(set(offsprings))

        if len(offsprings) % 2 != 0:
            offsprings.append(index_rank[winner])
        return offsprings
    
    def list_argsort(self, seq:list):

        return sorted(range(len(seq)), key=seq.__getitem__)
    
    def reg_weight(self, weight:float, mag:float):

        if weight > mag: weight = mag
        if weight < -mag: weight = -mag
        return weight

    def crossover_inplace(self, gene1:nn.Module, gene2:nn.Module):
        keys1 = list(gene1.state_dict())
        keys2 = list(gene2.state_dict())

        for key in keys1:
            if key not in keys2: continue

            W1 = gene1.state_dict()[key]
            W2 = gene2.state_dict()[key]

            if len(W1.shape) == 2:
                num_var = W1.shape[0]

                try:
                    num_crossovers = random.randint(0, int(num_var * 0.3))
                except:
                    num_crossovers = 1
                
                for i in range(num_crossovers):
                    receiver_choice = random.random()
                    ind_cr = random.randint(0, W1.shape[0] - 1)
                    if receiver_choice < 0.5:
                        W1[ind_cr, :] = W2[ind_cr, :]
                    else:
                        W2[ind_cr, :] = W1[ind_cr, :]
            
            elif len(W1.shape) == 1:
                if random.random() < 0.8: continue
                num_var = W1.shape[0]
                #Possibile inserire un numero di crossover variabili pure qua
                for i in range(1):
                    receiver_choice = random.random()
                    ind_cr = random.randint(0, W1.shape[0] - 1)
                    if receiver_choice < 0.5:
                        W1[ind_cr] = W2[ind_cr]
                    else:
                        W2[ind_cr] = W1[ind_cr]

    def mutate_inplace(self, gene:nn.Module):
        mut_strength = 0.1
        num_mut_frac = 0.05
        super_mut_strength = 10
        super_mut_prob = 0.05
        reset_prob = super_mut_prob + 0.02

        num_params = len(list(gene.parameters()))
        ssne_probabilities = np.random.uniform(0, 1, num_params) * 2

        for i, param in enumerate(gene.parameters()):

            W = param.data
            if len(W.shape) == 2:
                num_w = W.shape[0] * W.shape[1]
                ssne_prob = ssne_probabilities[i]

                if random.random() < ssne_prob:
                    num_mut = random.randint(0, int(math.ceil(num_mut_frac * num_w)))
                    for _ in range(num_mut):
                        ind_1 = random.randint(0, W.shape[0] - 1)
                        ind_2 = random.randint(0, W.shape[1] - 1)
                        random_num = random.random()
                        if random_num < super_mut_prob:
                            W[ind_1, ind_2] += random.gauss(0, super_mut_strength * W[ind_1, ind_2])
                        elif random_num < reset_prob:
                            W[ind_1, ind_2] = random.gauss(0, 0.1)
                        else:
                            W[ind_1, ind_2] += random.gauss(0, mut_strength * W[ind_1, ind_2])
                        W[ind_1, ind_2] = self.reg_weight(W[ind_1, ind_2], self.weight_clamp)
            
            elif len(W.shape) == 1:
                num_w = W.shape[0]
                ssne_prob = ssne_probabilities[i] * 0.04

                if random.random() < ssne_prob:
                    num_mut = random.randint(0, int(math.ceil(num_mut_frac * num_w)))
                
                    for _ in range(num_mut):
                        ind = random.randint(0, W.shape[0] - 1)
                        random_num = random.random()

                        if random_num < super_mut_prob:
                            W[ind] += random.gauss(0, super_mut_strength * W[ind])
                        elif random_num < reset_prob:
                            W[ind] += random.gauss(0, 1)
                        else:
                            W[ind] += random.gauss(0, mut_strength * W[ind])
                        
                        W[ind] = self.reg_weight(W[ind], self.weight_clamp)

    def reset_genome(self, gene:nn.Module):
        for param in (gene.parameters()):
            param.data.copy_(param.data)
    
    def roulette_wheel(self, probs:list, num_samples:int):
        probs = [prob - min(probs) + abs(min(probs)) for prob in probs]

        total_prob = sum(probs) if sum(probs) != 0 else 1.0

        probs = [prob / total_prob for prob in probs]

        out = []

        for _ in range(num_samples):
            rand = random.random()

            for i in range(len(probs)):
                if rand < sum(probs[0:i + 1]):
                    out.append(i)
                    break
        
        return out
    
    def evolve(self, pop:list[nn.Module], net_ind:list[int], fitness_evals:list[list], migration):
        
        self.gen += 1

        if isinstance(fitness_evals[0], list):
            for i in range(len(fitness_evals)):
                if self.ccea_red == "mean": fitness_evals[i] = sum(fitness_evals[i])/len(fitness_evals[i])
                elif self.ccea_red == "leniency": fitness_evals[i] = max(fitness_evals[i])
                elif self.ccea_red == "min": fitness_evals[i] = min(fitness_evals[i])
                else: sys.exit('Incorrect CCEA Reduction Scheme')

        lineage_score = []
        for ind, fitness in zip(net_ind, fitness_evals):
            self.lineage[ind].append(fitness)
            lineage_score.append(0.75 * sum(self.lineage[ind])/len(self.lineage[ind]) + 0.25 * fitness)
            if len(self.lineage[ind]) < self.lin_depth:
                self.lineage.pop(0)
        
        index_rank = self.list_argsort(fitness_evals)
        index_rank = index_rank.reverse()

        elitist_index = index_rank[:self.num_elites]

        lineage_rank = self.list_argsort(lineage_score[:])
        lineage_rank = lineage_rank.reverse()
        elitist_index = elitist_index + lineage_rank[:int(self.num_elites)]

        elitist_index = list(set(elitist_index))

        offspring = self.tournament(
            index_rank=index_rank,
            num_offspring=len(index_rank) - len(elitist_index) - len(migration),
            t_size=3,
        )

        elitist_index = [net_ind[i] for i in elitist_index]
        offspring = [net_ind[i] for i in offspring]

        unselected = []
        new_elite = []

        unselected = [i for i in range(len(pop)) if i not in offspring and i not in elitist_index]

        random.shuffle(unselected)
        
        for policy in migration:
            replacee = unselected.pop(0)
            hard_update(target=pop[replacee], source=policy)
            self.lineage[replacee] = [sum(lineage_score)/len(lineage_score)]

        for i in elitist_index:
            if len(unselected) >= 1: replacee = unselected.pop(0)
            elif len(offspring) >= 1: replacee = offspring.pop(0)
            else: continue

            new_elite.append(replacee)
            hard_update(target=pop[replacee], source=pop[i])

            self.lineage[replacee] = self.lineage[i][:]
        
        if len(unselected) % 2 != 0:
            unselected.append(unselected[random.randint(0, len(unselected) - 1)])
        for i, j in zip(unselected[0::2], unselected[1::2]):
            off_i = random.choice(new_elite)
            off_j = random.choice(offspring)
            hard_update(target=pop[i], source=pop[off_i])
            hard_update(target=pop[j], source=pop[off_j])

            self.crossover_inplace(pop[i], pop[j])

            self.lineage[i] = [0.5 * list_mean(self.lineage[off_i]) + 0.5 * list_mean(self.lineage[off_j])]
            self.lineage[j] = [0.5 * list_mean(self.lineage[off_i]) + 0.5 * list_mean(self.lineage[off_j])]
        
        for i, j in zip(offspring[0::2], offspring[1::2]):
            if random.random() < self.cross_prob:
                self.crossover_inplace(pop[i], pop[j])
                lin = 0.5 * list_mean(self.lineage[i]) + 0.5 * list_mean(self.lineage[j])
                self.lineage[i] = [lin]
                self.lineage[j] = [lin]
        
        for i in range(len(pop)):
            if i not in new_elite:
                if random.random() < self.mut_prob:
                    self.mutate_inplace(pop[i])
        
        self.all_offs[:] = offspring[:]
        return new_elite[0]