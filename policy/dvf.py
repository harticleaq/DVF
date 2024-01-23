import torch
import torch as th
from .network.basenet import RNN, Qjoint
from .network.qmixnet import QMixNet
from DVF.utils.utils import check

class POLICY:
    def __init__(self, args):
        self.args = args
        input_shape = self.args.obs_shape + self.args.n_agents + self.args.n_actions
        self.eval_rnn = RNN(input_shape, self.args).to(self.args.device)
        self.target_rnn = RNN(input_shape, self.args).to(self.args.device)


        self.Qjoint = Qjoint(args).to(self.args.device)
        self.target_Qjoint = Qjoint(args).to(self.args.device)



        self.mixnet = QMixNet(args).to(self.args.device)

        if self.args.load_model:
            map_location = 'cuda:0' if self.args.device == "cuda" else 'cpu'
            model_path  = self.args.model_path
            self.eval_rnn.load_state_dict(torch.load(model_path + f"/rnn_seed{self.args.seed}.pt"), map_location=map_location)
            self.Qjoint.load_state_dict(torch.load(model_path + f"/Qjoint_seed{self.args.seed}.pt") , map_location=map_location)
            self.mixnet.load_state_dict(torch.load(model_path + f"/mix_seed{self.args.seed}.pt"), map_location=map_location)
            print("load success!")
        
        #    th.save(self.mixnet.state_dict(), model_path + f"/mix_seed{self.args.seed}.pt")
        self.parameters = list(self.eval_rnn.parameters()) + list(self.Qjoint.parameters()) +\
                            list(self.mixnet.parameters())

        if self.args.double_Q:
            self.Qjoint2 = Qjoint(args).to(self.args.device)
            self.target_Qjoint2 = Qjoint(args).to(self.args.device)
            self.parameters += list(self.Qjoint2.parameters())
            self.target_Qjoint2.load_state_dict(self.Qjoint2.state_dict())


        if self.args.optimizer == "RMS":
            self.optimizer = th.optim.RMSprop(self.parameters, lr=self.args.lr)
        else:
            self.optimizer = th.optim.Adam(self.parameters, lr=self.args.lr)

        self.target_rnn.load_state_dict(self.eval_rnn.state_dict())
        self.target_Qjoint.load_state_dict(self.Qjoint.state_dict())

        self.eval_hidden = None
        self.target_hidden = None
        self.longdv = dict(dtype=th.long, device=self.args.device)

    def init_hidden(self, num):
        self.eval_hidden = th.zeros((num, self.args.n_agents, self.args.rnn_dim))
        self.target_hidden = th.zeros((num, self.args.n_agents, self.args.rnn_dim))

    def getQjoint(self, batch, hidden_eval, hidden_target, opt_onehot_target, hat=False):
        episode_num, max_episode_len, _, _ = batch['o'].shape
        states = batch['s']
        states_next = batch['s_next']
        u_onehot = batch['u_onehot']
        hidden_eval = hidden_eval.to(self.args.device)
        hidden_target = hidden_target.to(self.args.device)
        opt_onehot_target = opt_onehot_target.to(self.args.device)
        if hat:
            Q_evals = self.Qjoint(states, hidden_eval, opt_onehot_target)
            Q_targets = None
            Q_evals = Q_evals.view(episode_num, -1, 1)
            Q_evals = Q_evals.squeeze(-1)
        else:
            Q_evals = self.Qjoint(states, hidden_eval, u_onehot)
            Q_targets = self.target_Qjoint(states_next, hidden_target, opt_onehot_target)
            if self.args.double_Q:
                Q_targets2 = self.target_Qjoint2(states_next, hidden_target, opt_onehot_target)
                Q_targets = torch.minimum(Q_targets, Q_targets2)
            Q_evals = Q_evals.view(episode_num, -1, 1)
            Q_targets = Q_targets.view(episode_num, -1, 1)
            Q_evals = Q_evals.squeeze(-1)
            Q_targets = Q_targets.squeeze(-1)
        return Q_evals, Q_targets

    def get_inputs(self, batch, T):
        obs, obs_next= batch['o'][:, T], \
                                  batch['o_next'][:, T]
        u_onehot = batch['u_onehot'][:]
        episode_num = obs.shape[0]
        inputs, inputs_next = [], []
        inputs.append(obs)
        inputs_next.append(obs_next)
        if T == 0:
            inputs.append(th.zeros_like(u_onehot[:, T]))
        else:
            inputs.append(u_onehot[:, T - 1])
        inputs_next.append(u_onehot[:, T])

        inputs.append(th.eye(self.args.n_agents).unsqueeze(0).expand(episode_num, -1, -1).to(self.args.device))
        inputs_next.append(th.eye(self.args.n_agents).unsqueeze(0).expand(episode_num, -1, -1).to(self.args.device))

        inputs = th.cat([x.reshape(episode_num * self.args.n_agents, -1) for x in inputs], dim=1)
        inputs_next = th.cat([x.reshape(episode_num * self.args.n_agents, -1) for x in inputs_next], dim=1)
        return inputs, inputs_next

    def get_values(self, batch):
        B, T, N, C = batch['o'].shape
        q_evals, q_targets = [], []
        hidden_evals, hidden_targets = [], []
        for transition_idx in range(T):
            inputs, inputs_next = self.get_inputs(batch, transition_idx)  # 给obs加last_action、agent_id
            self.eval_hidden = self.eval_hidden.to(self.args.device)
            self.target_hidden = self.target_hidden.to(self.args.device)
            if transition_idx == 0:
                _, self.target_hidden = self.target_rnn(inputs, self.target_hidden)
            q_eval, self.eval_hidden = self.eval_rnn(inputs, self.eval_hidden)  # inputs维度为(40,96)，得到的q_eval维度为(40,n_actions)
            q_target, self.target_hidden = self.target_rnn(inputs_next, self.target_hidden)
            hidden_eval, hidden_target = self.eval_hidden.clone(), self.target_hidden.clone()
            hidden_eval = hidden_eval.view(B, self.args.n_agents, -1)
            hidden_target = hidden_target.view(B, self.args.n_agents, -1)
            hidden_evals.append(hidden_eval)
            hidden_targets.append(hidden_target)
            q_eval = q_eval.view(B, self.args.n_agents, -1)
            q_target = q_target.view(B, self.args.n_agents, -1)
            q_evals.append(q_eval)
            q_targets.append(q_target)
        hidden_evals = th.stack(hidden_evals, dim=1)
        hidden_targets = th.stack(hidden_targets, dim=1)
        q_evals = th.stack(q_evals, dim=1)
        q_targets = th.stack(q_targets, dim=1)
        return q_evals, q_targets, hidden_evals, hidden_targets

    def learn(self, batch, train_step):
        episode_num = batch['o'].shape[0]
        self.init_hidden(episode_num)
        for key in batch.keys():
            if key == 'u':
                batch[key] = check(batch[key]).to(**self.longdv)
            else:
                batch[key] = check(batch[key]).to(self.args.device)

        s, s_next, u, r, avail_u, avail_u_next, terminated = batch['s'], batch['s_next'], batch['u'], \
                                                             batch['r'], batch['avail_u'], batch['avail_u_next'], \
                                                             batch['terminated']
        mask = 1 - batch["padded"].float().squeeze(-1)

        q_evals, q_targets, hidden_eval, hidden_target = self.get_values(batch)

        individual_q_clone = q_evals.clone()
        individual_q_clone[avail_u == 0.0] = - 999999
        q_targets[avail_u_next == 0.0] = - 999999

        opt_onehot_eval = th.zeros(*individual_q_clone.shape)
        opt_action_eval = q_evals.argmax(dim=3, keepdim=True)
        opt_onehot_eval = opt_onehot_eval.scatter(-1, opt_action_eval[:, :].cpu(), 1)

        opt_onehot_target = th.zeros(*q_targets.shape)
        opt_action_target = q_targets.argmax(dim=3, keepdim=True)
        opt_onehot_target = opt_onehot_target.scatter(-1, opt_action_target[:, :].cpu(), 1)


        Qjoint, target_Qjoint = self.getQjoint(batch, hidden_eval, hidden_target, opt_onehot_target)

        y_dqn = r.squeeze(-1) + self.args.gamma * target_Qjoint * (1 - terminated.squeeze(-1))
        td_error = Qjoint - y_dqn.detach()
        l_td = ((td_error * mask) ** 2).sum() / mask.sum()

        Qjoint_opt, _ = self.getQjoint(batch, hidden_eval, hidden_target, opt_onehot_eval, hat=True)

        q_individual = th.gather(q_evals, dim=-1, index=u).squeeze(-1)
        q_opt = individual_q_clone.max(dim=-1)[0].squeeze(-1)
        d_q = q_individual - q_opt
        mix_q = self.mixnet(d_q, s).squeeze(-1)
        q_error = (mix_q) - (Qjoint - Qjoint_opt).detach()
        l_nopt = ((q_error * mask) ** 2).sum() / mask.sum()

        loss = l_td + l_nopt
        self.optimizer.zero_grad()
        loss.backward()
        clip_grad = th.nn.utils.clip_grad_norm_(self.parameters, self.args.grad_norm_clip)
        self.optimizer.step()

        if train_step > 0 and train_step % self.args.target_update_interval == 0:
            if self.args.hard_update:
                self.target_rnn.load_state_dict(self.eval_rnn.state_dict())
                self.target_Qjoint.load_state_dict(self.Qjoint.state_dict())
                if self.args.double_Q:
                    self.target_Qjoint2.load_state_dict(self.Qjoint2.state_dict())
            else:
                for target_param, param in zip(self.target_rnn.parameters(), self.eval_rnn.parameters()):
                    target_param.data.copy_((1 - self.args.lambda_soft_update) * target_param.data
                                            + self.args.lambda_soft_update * param.data)

                for target_param, param in zip(self.target_Qjoint.parameters(), self.Qjoint.parameters()):
                    target_param.data.copy_((1 - self.args.lambda_soft_update) * target_param.data
                                            + self.args.lambda_soft_update * param.data)


        return loss.item(), clip_grad.item(), Qjoint.sum().item(), y_dqn.sum().item()

    def save_model(self, model_path):
        th.save(self.eval_rnn.state_dict(), model_path + f"/rnn_seed{self.args.seed}.pt")
        th.save(self.Qjoint.state_dict(), model_path + f"/Qjoint_seed{self.args.seed}.pt")
        th.save(self.mixnet.state_dict(), model_path + f"/mix_seed{self.args.seed}.pt")