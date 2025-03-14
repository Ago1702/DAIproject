import random
import torch

import numpy as np

from torch.multiprocessing import Manager

class Buffer:
    '''
    Class Buffer: 
    '''

    def __init__(self, capacity, buffer_gpu):
        self.capacity = capacity
        self.buffer_gpu = buffer_gpu
        self.manager = Manager()
        self.tuples = self.manager.list()

        self.s = []
        self.ns = []
        self.a = []
        self.r = []
        self.done = []
        self.global_rew = []
        self.sT = None
        self.nsT = None
        self.aT = None
        self.rT = None
        self.doneT = None
        self.global_rewT = None

        self.pg_frames = 0
        self.total_frames = 0

    def data_filter(self, exp):
        self.s.append(exp[0])
        self.ns.append(exp[1])
        self.a.append(exp[2])
        self.r.append(exp[3])
        self.done.append(exp[4])
        self.global_rew.append(exp[5])
        
        self.pg_frames += 1
        self.total_frames += 1
    
    def refres(self):

        for _ in range(len(self.tuples)):
            exp = self.tuples.pop()
            self.data_filter(exp)
        
        while self.__len__() > self.capacity:
            self.s.pop(0)
            self.ns.pop(0)
            self.a.pop(0)
            self.r.pop(0)
            self.done.pop(0)
            self.global_rew.pop(0)
    
    def __len__(self):
        return len(self.s)
    
    def sample(self, batch_size:int):

        ind = random.sample(range(len(self.sT)), batch_size)

        return self.s[ind], self.ns[ind], self.a[ind], self.r[ind], self.done[ind], self.global_rew[ind]
    
    def tensorify(self):
        self.refres()

        if self.__len__() > 1:
            self.sT = torch.tensor(np.vstack(self.s))
            self.nsT = torch.tensor(np.vstack(self.ns))
            self.aT = torch.tensor(np.vstack(self.a))
            self.rT = torch.tensor(np.vstack(self.r))
            self.doneT = torch.tensor(np.vstack(self.done))
            self.global_rewT = torch.tensor(np.vstack(self.global_rew))
            if self.buffer_gpu:
                self.sT = self.sT.cuda()
                self.nsT = self.nsT.cuda()
                self.aT = self.aT.cuda()
                self.rT = self.rT.cuda()
                self.doneT = self.doneT.cuda()
                self.global_rewT = self.global_rewT.cuda()
            
        