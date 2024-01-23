import numpy as np
import torch as th
import importlib
from DVF.utils.utils import check

class MAC:
    def __init__(self, args):
        self.args = args

        alg = self.args.alg
        policy_module = importlib.import_module("DVF.policy." + alg)
        assert policy_module is not None
        self.policy = policy_module.POLICY(args)

    def choose_action(self, q_value, avail_actions, epsilon):
        avail_actions_ind = np.nonzero(avail_actions)[0]
        avail_actions = check(avail_actions).to(self.args.device).unsqueeze(0)
        q_value[avail_actions == 0.0] = - float("inf")
        if np.random.uniform() < epsilon:
            action = np.random.choice(avail_actions_ind)
        else:
            action = th.argmax(q_value)
        return action

    def choose_q(self, obs, last_action):
        inputs = np.array(obs.copy())
        inputs = np.hstack((inputs, last_action))
        inputs = np.hstack((inputs, np.eye(self.args.n_agents)))
        inputs = check(inputs).to(self.args.device)
        hidden_state = self.policy.eval_hidden.to(self.args.device)

        q_value, self.policy.eval_hidden = self.policy.eval_rnn(inputs, hidden_state)
        q_value = q_value.unsqueeze(0)
        return q_value

    def _get_max_episode_len(self, batch):
        terminated = batch['terminated']
        episode_num = terminated.shape[0]
        max_episode_len = 0
        for i in range(episode_num):
            for j in range(self.args.episode_limit):
                if terminated[i, j, 0] == 1:
                    if j + 1 >= max_episode_len:
                        max_episode_len = j + 1
                    break
        max_episode_len = self.args.max_episode_length if max_episode_len == 0 else max_episode_len
        return max_episode_len

    def train(self, batch, train_steps):
        max_episode_len = self._get_max_episode_len(batch)
        for key in batch.keys():
            batch[key] = batch[key][:, :max_episode_len]
        loss, clip_grad, q, target = self.policy.learn(batch, train_steps)
        return loss, clip_grad, q, target

    def save_model(self):
        self.policy.save_model(self.args.model_path)

