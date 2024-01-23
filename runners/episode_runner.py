import torch as th
import numpy as np

from DVF.envs import REGISTRY as ENV_REGISTRY
from DVF.componets import REGISTRY as CP_REGISTRY

class EpisodeRunner:
    def __init__(self, args, logger, config, ex_run):
        self.ex_run = ex_run
        self.config = config
        self.logger = logger
        self.args = args

        self.env = ENV_REGISTRY[self.args.env](**self.args.env_args)
        self.update_args()
        self.mac = CP_REGISTRY["mac"](self.args)
        self.buffer = CP_REGISTRY["buffer"](self.args)
        self.episode_limit = self.args.episode_limit
        self.epsilon = self.args.epsilon
        self.reduce_epsilon = (self.args.epsilon - self.args.min_epsilon) / self.args.epsilon_steps
        self.min_epsilon = self.args.min_epsilon

        self.losses = []


    def reset(self):
        self.env.reset()

    def update_args(self):
        env_info = self.get_env_info()
        self.args.n_actions = env_info["n_actions"]
        self.args.n_agents = env_info["n_agents"]
        self.args.state_shape = env_info["state_shape"]
        self.args.obs_shape = env_info["obs_shape"]
        self.args.episode_limit = env_info["episode_limit"]

    def get_env_info(self):
        return self.env.get_env_info()

    def get_epsilon(self, epsilon, evaluate=False):
        if evaluate:
            return 0
        return epsilon - self.reduce_epsilon if epsilon > self.min_epsilon else self.min_epsilon

    def generate_episode(self, evaluate=False, num=-1):
        step = 0
        terminated = False
        # if evaluate and num == 0:
        #     self.env.close()
        self.reset()
        o, u, r, s, avail_u, u_onehot, terminate, padded = [], [], [], [], [], [], [], []
        win_tag = False
        episode_reward = 0  # cumulative rewards
        last_action = np.zeros((self.args.n_agents, self.args.n_actions))
        epsilon = 0 if evaluate else self.epsilon

        self.mac.policy.init_hidden(1)
        while not terminated and step < self.episode_limit:
            obs = self.env.get_obs()
            state = self.env.get_state()
            actions, avail_actions, actions_onehot = [], [], []
            for agent_id in range(self.args.n_agents):
                avail_action = self.env.get_avail_agent_actions(agent_id)
                avail_actions.append(avail_action)
            avail_u.append(avail_actions)
            epsilon = self.get_epsilon(epsilon, evaluate)
            with th.no_grad():
                q_value = self.mac.choose_q(obs, last_action)
                for agent_id in range(self.args.n_agents):
                    action = self.mac.choose_action(q_value[:, agent_id], avail_actions[agent_id],
                                                               epsilon)

                    action_onehot = np.zeros(self.args.n_actions)
                    action_onehot[action] = 1
                    actions.append(np.int(action))
                    actions_onehot.append(action_onehot)
                    last_action[agent_id] = action_onehot
            reward, terminated, info = self.env.step(actions)
            win_tag = True if terminated and 'battle_won' in info and info['battle_won'] else False
            s.append(state)
            u.append(np.reshape(actions, [self.args.n_agents, 1]))
            u_onehot.append(actions_onehot)
            o.append(obs)
            r.append([reward])
            terminate.append([int(terminated)])
            padded.append([0.])
            episode_reward += reward
            step += 1

        obs = self.env.get_obs()
        state = self.env.get_state()
        o.append(obs)
        s.append(state)
        o_next = o[1:]
        s_next = s[1:]
        o = o[:-1]
        s = s[:-1]
        avail_actions = []
        for agent_id in range(self.args.n_agents):
            avail_action = self.env.get_avail_agent_actions(agent_id)
            avail_actions.append(avail_action)
        avail_u.append(avail_actions)
        avail_u_next = avail_u[1:]
        avail_u = avail_u[:-1]
        for i in range(step, self.episode_limit):
            o.append(np.zeros((self.args.n_agents, self.args.obs_shape)))
            u.append(np.zeros([self.args.n_agents, 1]))
            s.append(np.zeros(self.args.state_shape))
            r.append([0.])
            o_next.append(np.zeros((self.args.n_agents, self.args.obs_shape)))
            s_next.append(np.zeros(self.args.state_shape))
            u_onehot.append(np.zeros((self.args.n_agents, self.args.n_actions)))
            avail_u.append(np.zeros((self.args.n_agents, self.args.n_actions)))
            avail_u_next.append(np.zeros((self.args.n_agents, self.args.n_actions)))
            padded.append([1.])
            terminate.append([1.])

        episode = dict(o=o,
                       s=s,
                       u=u,
                       r=r,
                       avail_u=avail_u,
                       o_next=o_next,
                       s_next=s_next,
                       avail_u_next=avail_u_next,
                       u_onehot=u_onehot,
                       padded=padded,
                       terminated=terminate)

        if not evaluate:
            self.epsilon = epsilon
            self.buffer.store_episode(episode)
        return episode_reward, win_tag, step

    def evaluate(self):
        win_number = 0
        episode_rewards = 0
        for epoch in range(self.args.evaluate_pochs):
            episode_reward, win_tag, _ = self.generate_episode(evaluate=True, num=epoch)
            episode_rewards += episode_reward
            if win_tag:
                win_number += 1
        return win_number / self.args.evaluate_pochs, episode_rewards / self.args.evaluate_pochs

    def run(self, num):
        time_steps, train_steps, evaluate_steps = 0, 0, -1
        while time_steps < self.args.n_steps:
            print('Run {}, time_steps:{}'.format(num, time_steps))
            if time_steps // self.args.evaluate_interval > evaluate_steps and time_steps != 0:
                evaluate_steps += 1
                self.mac.save_model()
                win_rate, episode_reward = self.evaluate()
                print(f"Evaluate Win Rate:{win_rate}, Episode_reward:{episode_reward}")
                self.ex_run.log_scalar("evaluate/win_rate", win_rate, evaluate_steps)
                self.ex_run.log_scalar("evaluate/episode_reward", episode_reward, evaluate_steps)
            for i in range(self.args.n_episodes):
                _, _, step = self.generate_episode()
                time_steps += step
            # if time_steps // self.args.train_interval > train_steps and time_steps != 0:
            for train_step in range(self.args.train_steps):
                batch = self.buffer.sample(self.args.batch_size)
                loss, clip_norm, q, target = self.mac.train(batch, train_steps)
                self.ex_run.log_scalar("loss", loss, step=train_steps)
                self.ex_run.log_scalar("clip_norm", clip_norm, step=train_steps)
                self.ex_run.log_scalar("q", q, step=train_steps)
                self.ex_run.log_scalar("target", target, step=train_steps)
                train_steps += 1
        self.env.close()


