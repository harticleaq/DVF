import torch.nn as nn
import torch.nn.functional as f
import torch
import math


class RNN(nn.Module):
    # Because all the agents share the same network, input_shape=obs_shape+n_actions+n_agents
    def __init__(self, input_shape, args):
        super(RNN, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(input_shape, args.rnn_dim)
        self.rnn = nn.GRUCell(args.rnn_dim, args.rnn_dim)
        self.fc2 = nn.Linear(args.rnn_dim, args.n_actions)

    def forward(self, obs, hidden_state):
        x = f.relu(self.fc1(obs))
        h_in = hidden_state.reshape(-1, self.args.rnn_dim)
        h = self.rnn(x, h_in)
        # q = torch.abs(self.fc2(h))
        q = self.fc2(h)
        return q, h


class Qjoint(nn.Module):
    def __init__(self, args):
        super(Qjoint, self).__init__()
        self.args = args
        ae_dim = self.args.n_actions + self.args.rnn_dim
        self.hidden_action_encoding = nn.Sequential(nn.Linear(ae_dim, self.args.qjoint_dim),
                                             nn.ReLU(),
                                             nn.Linear(self.args.qjoint_dim, ae_dim))

        
        q_input = self.args.state_shape + self.args.n_actions + self.args.rnn_dim
        self.q = nn.Sequential(nn.Linear(q_input, self.args.qjoint_dim),
                               nn.ReLU(),
                               nn.Linear(self.args.qjoint_dim, self.args.qjoint_dim),
                               nn.ReLU(),
                               nn.Linear(self.args.qjoint_dim, 1),
                               )


    def forward(self, state, hidden, actions):  
        episode_num, max_episode_len, n_agents, _ = actions.shape
        hidden_actions = torch.cat([hidden, actions], dim=-1)
        hidden_actions = hidden_actions.reshape(-1, self.args.rnn_dim + self.args.n_actions)
        hidden_actions_encoding = self.hidden_action_encoding(hidden_actions)
        hidden_actions_encoding = hidden_actions_encoding.reshape(episode_num * max_episode_len, n_agents, -1)  # 变回n_agents维度用于求和
        hidden_actions_encoding = hidden_actions_encoding.sum(dim=-2)

        inputs = torch.cat([state.reshape(episode_num * max_episode_len, -1), hidden_actions_encoding], dim=-1)
        q = self.q(inputs)
        return q
