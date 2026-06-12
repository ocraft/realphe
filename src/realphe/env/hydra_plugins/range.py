from omegaconf import OmegaConf

OmegaConf.register_new_resolver('range', lambda x: list(range(0, x)))
