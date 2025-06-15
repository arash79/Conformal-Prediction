from embeddings.base import Embedding
import networkx as nx
import rdkit
import os
from karateclub import Graph2Vec
import pandas as pd
from rdkit import Chem
import yaml
from warnings import filterwarnings


filterwarnings('ignore')


with open("config.yaml", 'r') as file:
    config = yaml.safe_load(file)


class GraphRepresentation(Embedding):
    
    def __init__(
            self, 
            output_type: str = 'tabular',
            dimension_size: int = 128
            ):
        
        super().__init__()
        self.numeric_descriptors = self.dataset.copy()
        self.output_type = output_type
        self.dimension_size = dimension_size

    @staticmethod
    def __is_from_module(obj, module):
        target = module.__name__ if hasattr(module, '__name__') else module
        return obj.__class__.__module__.startswith(target)
    
    @staticmethod
    def __molecule_to_graph(molecule):

        graph = nx.Graph()

        for atom in molecule.GetAtoms():

            atom_attributes = list(filter(
                lambda attr: (('Has' in attr) or ('Get' in attr)) and ('Idx' not in attr), 
                dir(atom)
                ))
            
            attribute_values = {}

            for attribute in atom_attributes:
                try:
                    value = getattr(atom, attribute)()
                    if GraphRepresentation.__is_from_module(value, rdkit) is True:
                        pass
                    elif (isinstance(value, tuple) is True) or (isinstance(value, dict) is True):
                        pass
                    else:
                        attribute_values[attribute] = value
                except Exception as e:
                    pass

            graph.add_node(
                atom.GetIdx(),
                **attribute_values
                )
            
        for bond in molecule.GetBonds():

            bond_attributes = list(filter(
                lambda attr: (('Has' in attr) or ('Get' in attr)) and ('Idx' not in attr), 
                dir(bond)
                ))

            attribute_values = {}

            for attribute in bond_attributes:
                try:
                    value = getattr(atom, attribute)()
                    if GraphRepresentation.__is_from_module(value, rdkit) is True:
                        pass
                    elif (isinstance(value, tuple) is True) or (isinstance(value, dict) is True):
                        pass
                    else:
                        attribute_values[attribute] = value
                except Exception as e:
                    pass
            
            graph.add_edge(
                bond.GetBeginAtomIdx(),
                bond.GetEndAtomIdx(),
                **attribute_values
                )

        return graph
    
    def embed(self):

        if os.path.exists(config['MODIFIED_DATA'].format(output_type=f'{self.output_type}_{self.dimension_size}')):
            print('Already Exists!')
            self.numeric_descriptors = pd.read_pickle(config['MODIFIED_DATA'].format(output_type=f'{self.output_type}_{self.dimension_size}'))
            return self.numeric_descriptors

        self.numeric_descriptors['molecule'] = self.dataset['SMILES'].apply(Chem.MolFromSmiles)

        self.numeric_descriptors['graph'] = self.numeric_descriptors['molecule'].map(
            GraphRepresentation.__molecule_to_graph
            )
        
        if self.output_type == 'tabular': 
            
            model = Graph2Vec()
            model.fit(self.numeric_descriptors['graph'])
            graph2vec_representations = pd.DataFrame.from_records(
            model.get_embedding(),
            columns=[f'dim_{i}' for i in range(1, self.dimension_size + 1)]
            )
            self.numeric_descriptors = pd.concat(
                [self.numeric_descriptors, graph2vec_representations],
                axis=1
            )
            
        self.numeric_descriptors.to_pickle(config['MODIFIED_DATA'].format(output_type=f'{self.output_type}_{self.dimension_size}'))

        return self.numeric_descriptors
