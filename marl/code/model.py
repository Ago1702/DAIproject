import torch
import torch.nn as nn

class MultiHeadActor(nn.Module):

    def __init__(self, num_in:int, num_out:int, hidden:int = 2, heads:int = 1, mean:float = 0, std:float=0.4):
        super(MultiHeadActor, self).__init__()
        self.heads = heads
        self.num_out = num_out

        self.lin1 = nn.Linear(num_in, hidden)
        self.lin2 = nn.Linear(hidden, hidden)

        self.out = nn.Linear(hidden, num_out*heads)

        #Parametri rumore
        self.mean = mean
        self.std = std
        self.noise = torch.Tensor(num_out * heads)

        self.apply(weights_init)

    def forward(self, X:torch.Tensor) -> torch.Tensor:
        #Uso Tanh visto che spesso viene utilizzata in ambito di RL
        if(len(X.shape) < 2):
            X = X.unsqueeze(0)
        tmp = X
        tmp = torch.tanh(self.lin1(tmp))
        tmp = torch.tanh(self.lin2(tmp))
        return torch.tanh(self.out(tmp))

    def action(self, state:torch.Tensor, head:int=-1) -> torch.Tensor:
        X = self.forward(state)
        return self.head_select(X, head)
    
    def noise_action(self, state:torch.Tensor, head:int=-1) -> torch.Tensor:
        X = self.forward(state) + self.noise.normal_(self.mean, self.std)
        return self.head_select(X, head)
    
    def head_select(self, X:torch.Tensor, head:int=-1) -> torch.Tensor:
        if head != -1:
            start = head * self.num_out
            return X[:, start:start + self.num_out]
        return X

class QNetwork(nn.Module):
    def __init__(self, num_in:int, num_out:int, hidden_size:int):

        super(QNetwork, self).__init__()

        self.linear1 = nn.Linear(num_in + num_out, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 1)

        self.linear4 = nn.Linear(num_in + num_out, hidden_size)
        self.linear5 = nn.Linear(hidden_size, hidden_size)
        self.linear6 = nn.Linear(hidden_size, 1)

        self.apply(weights_init2)

    def forward(self, state:torch.Tensor, action:torch.Tensor):
        X = torch.cat([state, action], 1)
        x1 = X
        x1 = torch.tanh(self.linear1(x1))
        x1 = torch.tanh(self.linear2(x1))
        x1 = torch.tanh(self.linear3(x1))

        x2 = X
        x2 = torch.tanh(self.linear4(x2))
        x2 = torch.tanh(self.linear5(x2))
        x2 = torch.tanh(self.linear6(x2))

        return x1, x2

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        torch.nn.init.xavier_normal_(m.weight, gain=0.5)
        torch.nn.init.constant_(m.bias, 0)

def weights_init2(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        torch.nn.init.xavier_normal_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

'''
if __name__ == "__main__":
    actor = MultiHeadActor(5, 10, heads=3)
    state = torch.Tensor(5).normal_(0,1)
    print(state.shape)
    print(state)
    res = actor.action(state, 2)
    print(res)
    res = actor.action(state, 2)
    print(res)
    res = actor.noise_action(state, 2)
    print(res)
    res = actor(state)'''