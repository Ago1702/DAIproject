import torch
import torch.nn as nn

class MultiHeadActor(nn.Module):

    def __init__(self, num_in:int, num_out:int, hidden:int, heads:int):
        super(MultiHeadActor, self).__init__()
        self.heads = heads
        self.num_out = num_out

        self.lin1 = nn.Linear(num_in, hidden)
        self.lin2 = nn.Linear(hidden, hidden)

        self.out = nn.Linear(hidden, num_out*heads)

    def forward(self, X) -> torch.Tensor:
        tmp = X
        tmp = nn.ReLU(self.lin1(tmp))
        tmp = nn.ReLU(self.lin2(tmp))
        return nn.Sigmoid(self.out(tmp))

    def action(self, state:torch.Tensor, head:int=-1):
        X = self.forward(state)
        if head != -1:
            start = head * self.num_out
            return X[:, start:start + self.num_out]
        return X

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        torch.nn.init.xavier_normal_(m.weight, gain=0.5)
        torch.nn.init.constant_(m.bias, 0)