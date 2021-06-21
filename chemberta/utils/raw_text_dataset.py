import os

import numpy as np
import torch
from nlp import load_dataset
from rdkit import Chem
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator
from torch.utils.data import Dataset


class RawTextDataset(Dataset):
    """
    Custom Torch Dataset for tokenizing large (up to 100,000,000+ sequences) text corpuses,
    by not loading the entire dataset into cache and using lazy loading from disk (using huggingface's
    'NLP' library. See 'https://github.com/huggingface/nlp' for more details on the NLP package.
    Examples
    --------
    >>> from raw_text_dataset import RawTextDataset
    >>> dataset = RawTextDataset(tokenizer=tokenizer, file_path="shard_00_selfies.txt", block_size=512)
    Downloading: 100%
    1.52k/1.52k [00:03<00:00, 447B/s]
    Using custom data configuration default
    Downloading and preparing dataset text/default-f719ef2eb3ab586b (download: Unknown size, generated: Unknown size, post-processed: Unknown sizetotal: Unknown size) to /root/.cache/huggingface/datasets/text/default-f719ef2eb3ab586b/0.0.0/3a79870d85f1982d6a2af884fde86a71c771747b4b161fd302d28ad22adf985b...
    Dataset text downloaded and prepared to /root/.cache/huggingface/datasets/text/default-f719ef2eb3ab586b/0.0.0/3a79870d85f1982d6a2af884fde86a71c771747b4b161fd302d28ad22adf985b. Subsequent calls will reuse this data.
    Loaded Dataset
    Number of lines: 999988
    Block size: 512
    """

    def __init__(self, tokenizer, file_path: str, block_size: int):
        super().__init__()
        self.tokenizer = tokenizer
        self.file_path = file_path
        self.block_size = block_size

        data_files = get_data_files(file_path)
        self.dataset = load_dataset("text", data_files=data_files)["train"]
        print("Loaded Dataset")
        self.len = len(self.dataset)
        print("Number of lines: " + str(self.len))
        print("Block size: " + str(self.block_size))

    def __len__(self):
        return self.len

    def preprocess(self, feature_dict):
        batch_encoding = self.tokenizer(
            feature_dict["text"],
            add_special_tokens=True,
            truncation=True,
            max_length=self.block_size,
        )
        return torch.tensor(batch_encoding["input_ids"])

    def __getitem__(self, i):
        line = self.dataset[i]
        example = self.preprocess(line)
        return example


class RegressionTextIterable(torch.utils.data.IterableDataset):
    def __init__(self, tokenizer, file_path: str, block_size: int):
        super().__init__()
        print("Initializing dataset...")
        self.tokenizer = tokenizer
        self.file_path = file_path
        self.block_size = block_size

        print("Inferring CSV structure from first line...")
        self.dataset = load_dataset("text", data_files=get_data_files(file_path))[
            "train"
        ]
        self.num_labels = len(self.dataset[0]["text"].split(",")) - 1

        print("Loaded Dataset")
        self.len = len(self.dataset)
        print("Number of lines: " + str(self.len))
        print("Block size: " + str(self.block_size))

    def __iter__(self):
        for example in self.dataset:
            yield preprocess(example["text"], self.tokenizer, self.block_size)


class RegressionDataset(Dataset):
    def __init__(self, tokenizer, file_path: str, block_size: int):
        super().__init__()
        print("init dataset")
        self.tokenizer = tokenizer
        self.file_path = file_path
        self.block_size = block_size

        data_files = get_data_files(file_path)
        self.dataset = load_dataset("csv", data_files=data_files)["train"]
        dataset_columns = list(self.dataset.features.keys())
        self.smiles_column = dataset_columns[0]
        self.label_columns = dataset_columns[1:]
        self.num_labels = len(self.label_columns)

        print("Loaded Dataset")
        self.len = len(self.dataset)
        print("Number of lines: " + str(self.len))
        print("Block size: " + str(self.block_size))

    def __len__(self):
        return self.len

    def __getitem__(self, i):
        return preprocess(self.dataset[i]["text"], self.tokenizer, self.block_size)


class RegressionTextDataset(Dataset):
    def __init__(self, tokenizer, file_path: str, block_size: int):
        super().__init__()
        print("Initializing dataset...")
        self.tokenizer = tokenizer
        self.file_path = file_path
        self.block_size = block_size

        print("Inferring CSV structure from first line...")
        self.dataset = load_dataset("text", data_files=get_data_files(file_path))[
            "train"
        ]
        self.num_labels = len(self.dataset[0]["text"].split(",")) - 1
        print("Loaded Dataset")
        self.len = len(self.dataset)
        print("Number of lines: " + str(self.len))
        print("Block size: " + str(self.block_size))

    def __len__(self):
        return self.len

    def __getitem__(self, i):
        return preprocess(self.dataset[i]["text"], self.tokenizer, self.block_size)


def preprocess(line, tokenizer, block_size):
    def _clean_property(x):
        if x == "" or "inf" in x:
            return 0.0
        return float(x)

    line = line.split(",")
    smiles = line[0]
    labels = line[1:]

    batch_encoding = tokenizer(
        smiles,
        add_special_tokens=True,
        truncation=True,
        padding="max_length",
        max_length=block_size,
    )
    batch_encoding["label"] = [_clean_property(x) for x in labels]
    batch_encoding = {k: torch.tensor(v) for k, v in batch_encoding.items()}

    return batch_encoding


class LazyRegressionDataset(Dataset):
    """Computes RDKit properties on-the-fly."""

    def __init__(self, tokenizer, file_path: str, block_size: int):
        super().__init__()
        print("init dataset")
        self.tokenizer = tokenizer
        self.file_path = file_path
        self.block_size = block_size

        self.descriptors = [name for name, _ in Chem.Descriptors.descList]
        self.descriptors.remove("Ipc")
        self.calculator = MolecularDescriptorCalculator(self.descriptors)
        self.num_labels = len(self.descriptors)

        data_files = get_data_files(file_path)
        self.dataset = load_dataset("text", data_files=data_files)["train"]

        print("Loaded Dataset")
        self.len = len(self.dataset)
        print("Number of lines: " + str(self.len))
        print("Block size: " + str(self.block_size))

    def __len__(self):
        return self.len

    def _compute_descriptors(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            mol_descriptors = np.full(shape=(self.num_labels), fill_value=0.0)
        else:
            mol_descriptors = np.array(list(self.calculator.CalcDescriptors(mol)))
            mol_descriptors = np.nan_to_num(
                mol_descriptors, nan=0.0, posinf=0.0, neginf=0.0
            )
        assert mol_descriptors.size == self.num_labels

        return mol_descriptors

    def preprocess(self, feature_dict):
        smiles = feature_dict["text"]
        batch_encoding = self.tokenizer(
            smiles,
            add_special_tokens=True,
            truncation=True,
            padding="max_length",
            max_length=self.block_size,
        )
        batch_encoding = {k: torch.tensor(v) for k, v in batch_encoding.items()}

        mol_descriptors = self._compute_descriptors(smiles)
        batch_encoding["label"] = torch.tensor(mol_descriptors, dtype=torch.float32)

        return batch_encoding

    def __getitem__(self, i):
        feature_dict = self.dataset[i]
        example = self.preprocess(feature_dict)
        return example


def get_data_files(train_path):
    if os.path.isdir(train_path):
        return [
            os.path.join(train_path, file_name) for file_name in os.listdir(train_path)
        ]
    elif os.path.isfile(train_path):
        return train_path

    raise ValueError("Please pass in a proper train path")
