import yaml
import os

class Config:
    _instance = None

    def __new__(cls, config_path=None):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            if config_path:
                cls._instance.load_config(config_path)
        return cls._instance

    def __init__(self, config_path=None):
        if not hasattr(self, 'config_data'):  # Check if already initialized
            self.config_data = None
            self.config_path = config_path

    def load_config(self, config_path):
        """Load the YAML config file."""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found at {config_path}")
        
        with open(config_path, 'r') as file:
            try:
                self.config_data = yaml.safe_load(file)
            except yaml.YAMLError as e:
                raise ValueError(f"Error loading YAML file: {e}")
        
    def get(self, key, default=None):
        """Get a value from the config, with an optional default."""
        return self.config_data.get(key, default)

    def __str__(self):
        """Return a string representation of the config."""
        return str(self.config_data)
