from pathlib import Path
import yaml


def load_config(path, section=None):
    data = yaml.safe_load(Path(path).read_text())
    if not isinstance(data, dict):
        raise ValueError("Configuration must be a mapping")
    return data[section] if section else data
