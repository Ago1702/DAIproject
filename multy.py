from multiprocessing import Process
import os

import torch.multiprocessing.spawn
from torchrl.data import ReplayBuffer, ListStorage, LazyTensorStorage, LazyMemmapStorage, TensorDictReplayBuffer, PrioritizedReplayBuffer

from tensordict.tensordict import TensorDict

import torch

def info(title):
    print(title)
    print('module name:', __name__)
    print('parent process:', os.getppid())
    print('process id:', os.getpid())

def f(i, name, pipi, ciao, lol):
    len(lol)
    info('function f')
    print(pipi)
    for num in ciao:
        print(num)
    print('hello', name)

if __name__ == '__main__':
    lol = TensorDictReplayBuffer(storage=LazyMemmapStorage(100), shared=True)
    a = torch.rand((4, 8))
    b = torch.rand((4, 8))
    lol.add(TensorDict({'a':a}))
    lol.add(TensorDict({'a':b}))

    res = lol.sample(2)
    
    
    info('main line')
    torch.multiprocessing.spawn(f, args=('bob','lot', [1, 2, 3], lol))