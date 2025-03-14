from marl.code.lern_algo import MultiTD3
from torch.multiprocessing import Manager
from marl.code.model import MultiHeadActor
from marl.code.buffer import Buffer
from marl.code.evol import SSNE

class Agent:
    '''
    Class Agent: 
    '''

    def __init__(self, id, args):
        self.id = id
        self.args = args
        
        self.evolver = SSNE()

        self.manager = Manager()
        self.popn = self.manager.list()

        for _ in range(args.pop_size):
            self.popn.append(MultiHeadActor(args.state_dim, args.action_dim,
                                            args.hidden, args.config.num_agents))
            self.popn[-1].eval()
        
        self.algo = MultiTD3(
            id, 'TD3', args.state_dim, args.action_sim, args.hidden, args.actor_lr, args.critic_lr,
            args.gamma, args.tau, args.save_tag, args.folder, args.gpu, args.config.num_agents, args.init_w
        )

        self.rollout_actor = self.manager.list()
        self.rollout_actor.append(MultiHeadActor(args.state_dim, args.action_dim,
                                            args.hidden, args.config.num_agents))
        
        self.buffer = [Buffer(args.buffer_size, buffer_gpu=False) for _ in range(args.config.num_agents)]

        self.fitness = [[] for _ in range(args.pop_size)]

        self.champ_ind = 0

    def update_parameters(self):
        pass