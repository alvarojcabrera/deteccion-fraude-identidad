import yaml


def load_config(path="config.yaml"):
    """Lee config.yaml: la única fuente de las decisiones del proyecto."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)
