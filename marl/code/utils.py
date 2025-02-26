import torch
import torch.nn as nn

def init_weights(m:nn.Module):

    if type(m) == nn.Linear:
        nn.init.kaiming_uniform_(m.weight)
        m.bias.data.fill_(0.01)

def hard_update(target:nn.Module, source:nn.Module):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)
    
    try:
        target.wwid[0] = source.wwid[0]
    except:
        None

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def list_mean(l):
    if len(l) == 0: return None
    else:
        return sum(l) / len(l)
