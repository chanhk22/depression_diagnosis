# utils/config.py
import os, yaml

def _expand_env(value: str):
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, dict):
        return {k:_expand_env(v) for k,v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    return value

def load_config(path="configs/default.yaml"):
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return _expand_env(cfg)
