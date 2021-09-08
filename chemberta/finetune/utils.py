from collections import OrderedDict
from dataclasses import dataclass
import os

import pandas as pd
import numpy as np

import torch
from typing import Dict, List, Optional

from transformers import RobertaTokenizerFast

@dataclass
class FinetuneDatasets:
    train_dataset: str
    valid_dataset: torch.utils.data.Dataset
    valid_dataset_unlabeled: torch.utils.data.Dataset
    test_dataset: torch.utils.data.Dataset
    num_labels: int
    norm_mean: List[float]
    norm_std: List[float]


class FinetuneDataset(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer, include_labels=True):

        self.encodings = tokenizer(df["smiles"].tolist(), truncation=True, padding=True)
        self.labels = df.iloc[:, 1].values
        self.include_labels = include_labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.include_labels and self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])


def prune_state_dict(model_dir: str) -> Dict:
    """Remove problematic keys from state dictionary

    Args:
        model_dir: local model directory

    Returns:
        new_state_dict: torch state dictionary
    """
    if not (model_dir and os.path.exists(os.path.join(model_dir, "pytorch_model.bin"))):
        return None

    state_dict_path = os.path.join(model_dir, "pytorch_model.bin")
    assert os.path.exists(
        state_dict_path
    ), f"No `pytorch_model.bin` file found in {model_dir}"
    loaded_state_dict = torch.load(state_dict_path)
    state_keys = loaded_state_dict.keys()
    keys_to_remove = [
        k for k in state_keys if k.startswith("regression") or k.startswith("norm")
    ]

    new_state_dict = OrderedDict({**loaded_state_dict})
    for k in keys_to_remove:
        del new_state_dict[k]
    return new_state_dict


def get_latest_checkpoint(saved_model_dir) -> List[str]:
    """Get the folder for the latest checkpoint

    Args:
        saved_model_dir: directory with checkpoints

    Returns:
        latest_checkpoint_dir: subdir with the latest checkpoint
    """
    iters = [
        int(x.split("-")[-1]) for x in os.listdir(saved_model_dir) if "checkpoint" in x
    ]
    iters.sort()
    latest_checkpoint_dir = os.path.join(saved_model_dir, f"checkpoint-{iters[-1]}")
    return latest_checkpoint_dir


def get_finetune_datasets(
    dataset_name: str,
    tokenizer: RobertaTokenizerFast,
    is_molnet: bool,
    split: Optional[str] = "scaffold",
) -> FinetuneDatasets:
    """Fetch data and turn into FinetuneDatasets

    Args:
        dataset_name: name of molnet dataset or local dataset dir
        tokenizer: tokenizer object
        is_molnet: whether or not the dataset is a molnet dataset
        split: type of split to use for DeepChem data loader
    """
    if is_molnet:
        try:
            from chemberta.utils.molnet_dataloader import load_molnet_dataset
        except ImportError:
            raise ImportError("`deepchem` needed to load MolNet datasets")

        tasks, (train_df, valid_df, test_df), _ = load_molnet_dataset(
            dataset_name, split=split, df_format="chemprop"
        )
        assert len(tasks) == 1
    else:
        train_df = pd.read_csv(os.path.join(dataset_name, "train.csv"))
        valid_df = pd.read_csv(os.path.join(dataset_name, "valid.csv"))
        test_df = pd.read_csv(os.path.join(dataset_name, "test.csv"))

    train_dataset = FinetuneDataset(train_df, tokenizer)
    valid_dataset = FinetuneDataset(valid_df, tokenizer)
    valid_dataset_unlabeled = FinetuneDataset(valid_df, tokenizer, include_labels=False)
    test_dataset = FinetuneDataset(test_df, tokenizer, include_labels=False)

    num_labels = len(np.unique(train_dataset.labels))
    norm_mean = [np.mean(np.array(train_dataset.labels), axis=0)]
    norm_std = [np.std(np.array(train_dataset.labels), axis=0)]

    return FinetuneDatasets(
        train_dataset,
        valid_dataset,
        valid_dataset_unlabeled,
        test_dataset,
        num_labels,
        norm_mean,
        norm_std,
    )
