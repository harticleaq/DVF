from functools import partial
from smac.env import StarCraft2Env, MultiAgentEnv
REGISTRY = {}

def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)

REGISTRY["sc2"] = partial(env_fn, env=StarCraft2Env)


