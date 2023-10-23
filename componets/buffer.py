import numpy as np
import threading

class ReplayBuffer:
    def __init__(self, args):
        self.args = args
        self.size = self.args.buffer_size
        self.episode_limit = self.args.episode_limit
        self.current_index = 0
        self.current_size = 0

        self.buffers = {
            'o': np.empty([self.size, self.episode_limit, self.args.n_agents, self.args.obs_shape]),
            'u': np.empty([self.size, self.episode_limit, self.args.n_agents, 1]),
            's': np.empty([self.size, self.episode_limit, self.args.state_shape]),
            'r': np.empty([self.size, self.episode_limit, 1]),
            'o_next': np.empty([self.size, self.episode_limit, self.args.n_agents, self.args.obs_shape]),
            's_next': np.empty([self.size, self.episode_limit, self.args.state_shape]),
            'avail_u': np.empty([self.size, self.episode_limit, self.args.n_agents, self.args.n_actions]),
            'avail_u_next': np.empty([self.size, self.episode_limit, self.args.n_agents, self.args.n_actions]),
            'u_onehot': np.empty([self.size, self.episode_limit, self.args.n_agents, self.args.n_actions]),
            'padded': np.empty([self.size, self.episode_limit, 1]),
            'terminated': np.empty([self.size, self.episode_limit, 1])
        }
        self.lock = threading.Lock()

    def sample(self, batch_size):
        temp = {}
        max_len = min(self.current_size, self.args.buffer_size)
        index = np.random.randint(0, max_len, batch_size)
        for key in self.buffers.keys():
            temp[key] = self.buffers[key][index]
        return temp

    def store_episode(self, episode):
        with self.lock:
            idxs = self.current_index % self.args.buffer_size
            self.current_index += 1
            self.current_size += 1
            self.buffers['o'][idxs] = episode['o']
            self.buffers['u'][idxs] = episode['u']
            self.buffers['s'][idxs] = episode['s']
            self.buffers['r'][idxs] = episode['r']
            self.buffers['o_next'][idxs] = episode['o_next']
            self.buffers['s_next'][idxs] = episode['s_next']
            self.buffers['avail_u'][idxs] = episode['avail_u']
            self.buffers['avail_u_next'][idxs] = episode['avail_u_next']
            self.buffers['u_onehot'][idxs] = episode['u_onehot']
            self.buffers['padded'][idxs] = episode['padded']
            self.buffers['terminated'][idxs] = episode['terminated']

