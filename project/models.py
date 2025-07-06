import torch
import torch.nn as nn
import torch.nn.functional as F


'''The initialization policies are copied from the original paper'''

def weights_init_policy_fn(m:nn.Module):
    class_name = m.__class__.__name__
    if class_name.find('Linear') != -1:
        torch.nn.init.xavier_uniform_(m.weight, gain=0.5)
        torch.nn.init.constant_(m.bias, 0)

def weights_init_value_fn(m:nn.Module):
    class_name = m.__class__.__name__
    if class_name.find('Linear') != -1:
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

class TeamNet(nn.Module):
    def __init__(self, num_heads:int, num_inputs:int, num_actions:int, hidden_size:int=64, hidden_layer:int=2):
        super(TeamNet, self).__init__()

        self.num_heads = num_heads
        self.num_inputs = num_inputs
        self.num_actions = num_actions
        self.hidden_size = hidden_size
        self.hidden_layer = hidden_layer
        self.eps = 0.95


        self.input = nn.Linear(num_inputs, hidden_size)

        hidden = [nn.ReLU()]
        for i in range(hidden_layer):
            hidden.append(nn.Linear(hidden_size, hidden_size))
            hidden.append(nn.ReLU())
        
        self.hidden = nn.Sequential(*hidden)

        self.actions = nn.Linear(hidden_size, num_heads*num_actions)

        self.register_buffer("noise", torch.zeros(num_actions * num_heads))

        self.apply(weights_init_policy_fn)

    #TODO: Add support for different agent

    def forward(self, X:torch.Tensor) -> torch.Tensor:
        X = self.input.forward(X)
        X = self.hidden.forward(X)
        X = self.actions.forward(X)
        return F.sigmoid(X)
    
    def head_select(self, X:torch.Tensor, head) -> torch.Tensor:
        if head == -1:
            return X
        else:
            start = head * self.num_actions
            if len(X.shape) < 2:
                return X[start:start + self.num_actions]
            return X[:,  start:start + self.num_actions]
    
    def action(self, X:torch.Tensor, head:int=-1) -> torch.Tensor:
        X = self.forward(X)
        X = self.head_select(X, head)
        return X

    def noise_action(self, X:torch.Tensor, head:int=-1) -> torch.Tensor:
        X = (1 - self.eps) * self.forward(X) + self.eps * self.noise.normal_(0., 0.4)
        X = self.head_select(X, head) 
        return X

        
class CriticNet(nn.Module):

    def __init__(self, state_size:int, actions_size:int, hidden_size:int=64, hidden_layer:int=2):
        super(CriticNet, self).__init__()
        self.state_size = state_size
        self.actions_size = actions_size
        self.hidden_size = hidden_size
        self.hidden_layer = hidden_layer

        net1 = [nn.Linear(state_size + actions_size, hidden_size)]

        net1.append(nn.ReLU())
        for i in range(hidden_layer):
            net1.append(nn.Linear(hidden_size, hidden_size))
            net1.append(nn.ReLU())

        net1.append(nn.Linear(hidden_size, 1))
        self.net1 = nn.Sequential(*net1)

        net2 = [nn.Linear(state_size + actions_size, hidden_size)]

        net2.append(nn.ReLU())
        for i in range(hidden_layer):
            net2.append(nn.Linear(hidden_size, hidden_size))
            net2.append(nn.ReLU())

        net2.append(nn.Linear(hidden_size, 1))
        self.net2 = nn.Sequential(*net2)

    def forward(self, observation, actions) -> torch.Tensor:
        X = torch.cat([observation, actions], 1)
        X1 = self.net1.forward(X)
        X2 = self.net2.forward(X)
        return X1, X2
    
'''if __name__ == '__main__':
    net = CriticNet(5, 5)
    nac = TeamNet(3, 5, 5)
    print(net)
    obs = torch.zeros((1, 5)).normal_()
    act = torch.zeros((1, 5)).normal_()
    print(net.forward(obs, act))
    print(nac.action(obs))
    print(nac.action(obs, 0))
    print(nac.action(obs, 1))
    print(nac.action(obs, 2))'''