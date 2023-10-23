REGISTRY = {}
from .mac import MAC
REGISTRY["mac"] = MAC

from .buffer import ReplayBuffer
REGISTRY["buffer"] = ReplayBuffer
