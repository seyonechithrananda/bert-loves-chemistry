"""Computes means and stds of RDKit descriptors on a .smi file, needed for MTR pretraining."""

import json
import os
import pandas as pd
from tqdm import tqdm

import numpy as np
from absl import app, flags
from rdkit import Chem
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator

flags.DEFINE_string(name="smiles_file", default=None, help="")
flags.DEFINE_string(name="output_file", default="normalization_values.json", help="")


FLAGS = flags.FLAGS


def main(argv):

    descriptors = [name for name, _ in Chem.Descriptors.descList]
    descriptors.remove("Ipc")
    calculator = MolecularDescriptorCalculator(descriptors)
    
    df = pd.read_csv(FLAGS.smiles_file, names=["smiles"])
    computed_descriptors = np.vstack([compute_descriptors(s, calculator, len(descriptors)) for s in tqdm(df["smiles"])])
    
    d = {
        "mean": list(np.mean(computed_descriptors, axis=0)),
        "std": list(np.std(computed_descriptors, axis=0)),
    }
    
    with open(FLAGS.output_file, "w") as f:
        json.dump(d, f)


def compute_descriptors(smiles, calculator, n_descriptors):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        mol_descriptors = np.full(shape=(n_descriptors), fill_value=0.0)
    else:
        mol_descriptors = np.array(list(calculator.CalcDescriptors(mol)))
        mol_descriptors = np.nan_to_num(mol_descriptors, nan=0.0, posinf=0.0, neginf=0.0)
    return mol_descriptors
    

if __name__ == "__main__":
    app.run(main=main)