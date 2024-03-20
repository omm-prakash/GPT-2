import yaml

class DictToObject:
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            setattr(self, key, value)

def load_config(file_path):
    with open(file_path, 'r') as stream:
        try:
            config_data = yaml.safe_load(stream)
            return DictToObject(config_data)
        except yaml.YAMLError as exc:
            print(f"Error loading YAML file: {exc}")
            return None

class Error(Exception):
    def __init__(self, message):
        super().__init__(message)