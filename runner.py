import pprint

from types import SimpleNamespace as SN
from runners import REGISTRY as RUNNER_REGISTRY

def run(_run, _config, _log):
    args = SN(**_config)
    _log.info("Experiment Parameters:")

    experiment_params = pprint.pformat(_config, indent=2, width=1)
    _log.info("\n\n" + experiment_params + "\n")

    runner = RUNNER_REGISTRY[args.runner](args, _log, _config, _run)

    runner.run(args.seed)
