import torch as th

def check(x):
    x = th.tensor(x, dtype=th.float32)
    return x

def _t2n(x):
    x = x.detach().numpy().cpu()
    return x