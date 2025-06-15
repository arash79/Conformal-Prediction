from abc import ABC, abstractmethod
import pandas as pd
import yaml


with open("config.yaml", 'r') as file:
    config = yaml.safe_load(file)


class Embedding(ABC):

    def __init__(self):
        super().__init__()
        self.dataset = pd.read_csv(config['DATA']).dropna(axis=0)
    
    @abstractmethod 
    def embed(self):
        pass
