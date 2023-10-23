import torch as th
import numpy as np

from QLMIX_trans.envs import REGISTRY as ENV_REGISTRY
from QLMIX_trans.componets import REGISTRY as CP_REGISTRY

from multiprocessing import Pipe, Process
from functools import partial

class CloudpickleWrapper():
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """
    def __init__(self, x):
        self.x = x
    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)
    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)

def env_worker(remote, env_fn):
    env = env_fn.x()
    while True:
        cmd, data = remote.recv()
        if cmd == "step":
            actions = data
            # Take a step in the environment
            reward, terminated, env_info = env.step(actions)
            # Return the observations, avail_actions and state to make the next action
            state = env.get_state()
            avail_actions = env.get_avail_actions()
            obs = env.get_obs()
            remote.send({
                # Data for the next timestep needed to pick an action
                "state": state,
                "avail_actions": avail_actions,
                "obs": obs,
                # Rest of the data for the current timestep
                "reward": reward,
                "terminated": terminated,
                "info": env_info
            })
        elif cmd == "reset":
            env.reset()
            remote.send({
                "state": env.get_state(),
                "avail_actions": env.get_avail_actions(),
                "obs": env.get_obs()
            })
        elif cmd == "close":
            env.close()
            remote.close()
            break
        elif cmd == "get_env_info":
            remote.send(env.get_env_info())
        elif cmd == "get_stats":
            remote.send(env.get_stats())
        else:
            raise NotImplementedError

class ParallelRunner:
    def __init__(self, args):
        self.args = args
        self.num_thread = self.args.num_thread


        self.parent_conns, self.worker_conns = zip(*[Pipe() for _ in range(self.num_thread)])
        env_fn = ENV_REGISTRY[self.args.env]
        self.ps = [Process(target=env_worker, args=(worker_conn,\
                        CloudpickleWrapper(partial(env_fn, **self.args.env_args))))
                        for worker_conn in self.worker_conns]

        for p in self.ps:
            p.daemon = True
            p.start()

        self.parent_conns[0].send(("get_env_info", None))
        self.env_info = self.parent_conns[0].recv()
        self.episode_limit = self.env_info["episode_limit"]

        self.update_args()
        self.mac = CP_REGISTRY["mac"](self.args)
        self.buffer = CP_REGISTRY["buffer"](self.args)
        self.epsilon = self.args.epsilon
        self.reduce_epsilon = (self.args.epsilon - self.args.min_epsilon) / self.args.epsilon_steps
        self.min_epsilon = self.args.min_epsilon

        self.step = 0
        self.step_env = 0

    def update_args(self):
        env_info = self.get_env_info()
        self.args.n_actions = env_info["n_actions"]
        self.args.n_agents = env_info["n_agents"]
        self.args.state_shape = env_info["state_shape"]
        self.args.obs_shape = env_info["obs_shape"]
        self.args.episode_limit = env_info["episode_limit"]

    def get_env_info(self):
        return self.env_info

    def reset(self):
        for parent_conn in self.parent_conns:
            parent_conn.send(("reset", None))

        pre_transition_data = {
            "state": [],
            "avail_actions": [],
            "obs": []
        }
        for parent_conn in self.parent_conns:
            data = parent_conn.recv()
            pre_transition_data["state"].append(data["state"])
            pre_transition_data["avail_actions"].append(data["avail_actions"])
            pre_transition_data["obs"].append(data["obs"])

        self.step = 0

    def run(self):
        self.reset()

        terminated = False

        self.mac.policy.init_hidden(self.num_thread)

