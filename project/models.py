import torch
import torch.nn as nn

class Actor(nn.Module):
    '''
    Class Actor: Basic actor class for a first test
    '''

    def __init__(self, num_input:int, num_output:int, num_hidden:int=1, hidden_size:int=256, mean:float=0, std:float=0.4):
        super(Actor, self).__init__()
        self.num_input = num_input
        self.num_output = num_output
        self.num_hidden = num_hidden
        self.hidden_size = hidden_size
        self.mean = mean
        self.std = std
        self.noise = torch.Tensor(self.num_output)
        
        self.input_layer = nn.Linear(self.num_input, self.hidden_size)
        hidden_list = [nn.LeakyReLU()]
        for i in range(self.num_hidden):
            hidden_list.append(nn.Linear(self.hidden_size, self.hidden_size))
            hidden_list.append(nn.LeakyReLU())
        self.hidden_layer = nn.Sequential(*hidden_list)

        self.output_layer = nn.Linear(self.hidden_size, self.num_output)
    
    def forward(self, X:torch.Tensor):
        if len(X.shape) < 2:
            X = X.unsqueeze(0)
        tmp = X
        tmp = self.input_layer(tmp)
        tmp = self.hidden_layer(tmp)
        tmp = self.output_layer(tmp)
        return torch.tanh(tmp)
    
    def noisy_forward(self, X:torch.Tensor):
        tmp = self.forward(X) + self.noise.normal_(self.mean, self.std)
        return tmp


class Critic(nn.Module):
    '''
    Class Critic: A simple critic network
    '''

    def __init__(self, num_input:int, num_output:int, num_hiddem:int=1, hidden_size:int=256):
        super(Critic, self).__init__()

        self.num_input = num_input
        self.num_output = num_output
        self.num_hidden = num_hiddem
        self.hidden_size = hidden_size

        self.input_layer = nn.Linear(self.num_input, self.hidden_size)
        hidden_list = [nn.LeakyReLU()]
        for i in range(self.num_hidden):
            hidden_list.append(nn.Linear(self.hidden_size, self.hidden_size))
            hidden_list.append(nn.LeakyReLU())
        self.hidden_layer = nn.Sequential(*hidden_list)
        self.output_layer = nn.Linear(self.hidden_size, 1)
    
    def forward(self, X:torch.Tensor):
        if len(X.shape) < 2:
            X = X.unsqueeze(0)
        tmp = X
        tmp = self.input_layer(tmp)
        tmp = self.hidden_layer(tmp)
        tmp = self.output_layer(tmp)
        return torch.tanh(tmp)

        
        