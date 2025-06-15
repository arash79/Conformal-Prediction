import pandas as pd
from embeddings.base import Embedding
from rdkit.Chem import rdFingerprintGenerator
from rdkit import Chem


class FingerPrints(Embedding):

    def __init__(
            self, 
            fingerprint_type: str = 'rdk', 
            fingerprint_size: int = 2048
            ):
        
        super().__init__()
        self.numeric_descriptors = self.dataset.copy()
        self.fingerprint_type = fingerprint_type.lower()
        self.fingerprint_size = fingerprint_size

    def __molecule_to_fingerprint(self, molecule):

        if self.fingerprint_type == 'morgan':
            fingerprint_generator = rdFingerprintGenerator.GetMorganGenerator(fpSize=self.fingerprint_size)
        elif self.fingerprint_type == 'rdk':
            fingerprint_generator = rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=self.fingerprint_size)
        elif self.fingerprint_type == 'atom_pair':
            fingerprint_generator = rdFingerprintGenerator.GetAtomPairGenerator(fpSize=self.fingerprint_size)
        elif self.fingerprint_type == 'topological_torsion':
            fingerprint_generator = rdFingerprintGenerator.GetTopologicalTorsionGenerator(fpSize=self.fingerprint_size)

        return fingerprint_generator.GetFingerprintAsNumPy(molecule)

    def embed(self):

        self.numeric_descriptors['molecule'] = self.dataset['SMILES'].apply(Chem.MolFromSmiles)

        rdk_fingerprints = pd.DataFrame.from_records(
            self.numeric_descriptors['molecule'].map(
                self.__molecule_to_fingerprint
                ),
            columns=[f'{self.fingerprint_type}_descriptor_{i}' for i in range(1, self.fingerprint_size + 1)]
        )

        self.numeric_descriptors = pd.concat(
            [self.numeric_descriptors, 
            rdk_fingerprints],
            axis=1
        )

        return self.numeric_descriptors
    
    def get_data(self):
        return self.numeric_descriptors
    