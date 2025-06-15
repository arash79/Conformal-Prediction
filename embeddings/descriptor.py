from embeddings.base import Embedding
from typing import List
from rdkit.ML.Descriptors import MoleculeDescriptors # type: ignore
from rdkit.Chem import Descriptors
from rdkit import Chem
import pandas as pd


class RDKDescriptors(Embedding):

    def __init__(self, descriptors: List[str] = None):

        super().__init__()
        self.numeric_descriptors = self.dataset.copy()

        if descriptors is None:
            descriptors_list = MoleculeDescriptors.MolecularDescriptorCalculator(x[0] for x in Descriptors._descList)
            self.descriptors = descriptors_list.GetDescriptorNames()
        else:
            self.descriptors = descriptors
        
        self.descriptor_functions = MoleculeDescriptors.MolecularDescriptorCalculator(
            self.descriptors
        )
    
    def __calculate_description(self, molecule):
        return self.descriptor_functions.CalcDescriptors(molecule)
    
    def embed(self):

        self.numeric_descriptors['molecule'] = self.dataset['SMILES'].apply(Chem.MolFromSmiles)
        
        descriptors = pd.DataFrame.from_records(
            self.numeric_descriptors['molecule'].map(
                self.__calculate_description
                ),
            columns=self.descriptors
        )

        self.numeric_descriptors = pd.concat(
            [self.numeric_descriptors, 
            descriptors],
            axis=1
        )

        return self.numeric_descriptors

    def get_data(self):
        return self.numeric_descriptors
    