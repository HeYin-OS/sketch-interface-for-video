import yaml


def read_yaml(filename='configs/applicationContext.yml'):
    with open(filename, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg

