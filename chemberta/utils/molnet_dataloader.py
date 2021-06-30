import os
from typing import List

import pandas as pd
from deepchem.molnet import *
from rdkit import Chem

MOLNET_DIRECTORY = {
    "bace_classification": {
        "dataset_type": "classification",
        "load_fn": load_bace_classification,
        "split": "scaffold",
    },
    "bace_regression": {
        "dataset_type": "regression",
        "load_fn": load_bace_regression,
        "split": "scaffold",
    },
    "bbbp": {
        "dataset_type": "classification",
        "load_fn": load_bbbp,
        "split": "scaffold",
    },
    "clearance": {
        "dataset_type": "regression",
        "load_fn": load_clearance,
        "split": "scaffold",
    },
    "clintox": {
        "dataset_type": "classification",
        "load_fn": load_clintox,
        "split": "scaffold",
        "tasks_wanted": ["CT_TOX"],
    },
    "delaney": {
        "dataset_type": "regression",
        "load_fn": load_delaney,
        "split": "scaffold",
    },
    "hiv": {
        "dataset_type": "classification",
        "load_fn": load_hiv,
        "split": "scaffold",
    },
    # pcba is very large and breaks the dataloader
    #     "pcba": {
    #         "dataset_type": "classification",
    #         "load_fn": load_pcba,
    #         "split": "scaffold",
    #     },
    "lipo": {
        "dataset_type": "regression",
        "load_fn": load_lipo,
        "split": "scaffold",
    },
    "qm7": {
        "dataset_type": "regression",
        "load_fn": load_qm7,
        "split": "random",
    },
    "qm8": {
        "dataset_type": "regression",
        "load_fn": load_qm8,
        "split": "random",
    },
    "qm9": {
        "dataset_type": "regression",
        "load_fn": load_qm9,
        "split": "random",
    },
    "sider": {
        "dataset_type": "classification",
        "load_fn": load_sider,
        "split": "scaffold",
    },
    "tox21": {
        "dataset_type": "classification",
        "load_fn": load_tox21,
        "split": "scaffold",
        "tasks_wanted": ["SR-p53"],
    },
}


def get_dataset_info(name: str):
    return MOLNET_DIRECTORY[name]


def load_molnet_dataset(
    name: str,
    split: str = None,
    tasks_wanted: List = None,
    df_format: str = "chemberta",
):
    """Loads a MolNet dataset into a DataFrame ready for either chemberta or chemprop.

    Args:
        name: Name of MolNet dataset (e.g., "bbbp", "tox21").
        split: Split name. Defaults to the split specified in MOLNET_DIRECTORY.
        tasks_wanted: List of tasks from dataset. Defaults to `tasks_wanted` in MOLNET_DIRECTORY, if specified, or else all available tasks.
        df_format: `chemberta` or `chemprop`

    Returns:
        tasks_wanted, (train_df, valid_df, test_df), transformers

    """
    load_fn = MOLNET_DIRECTORY[name]["load_fn"]
    tasks, splits, transformers = load_fn(
        featurizer="Raw", split=split or MOLNET_DIRECTORY[name]["split"]
    )

    # Default to all available tasks
    if tasks_wanted is None:
        tasks_wanted = MOLNET_DIRECTORY[name].get("tasks_wanted", tasks)
    print(f"Using tasks {tasks_wanted} from available tasks for {name}: {tasks}")

    return (
        tasks_wanted,
        [
            make_dataframe(
                s,
                MOLNET_DIRECTORY[name]["dataset_type"],
                tasks,
                tasks_wanted,
                df_format,
            )
            for s in splits
        ],
        transformers,
    )


def write_molnet_dataset_for_chemprop(
    name: str, split: str = None, tasks_wanted: List = None, data_dir: str = None
):
    """Writes a MolNet dataset to separate train, val, test CSVs ready for chemprop.

    Args:
        name: Name of MolNet dataset (e.g., "bbbp", "tox21").
        split: Split name. Defaults to the split specified in MOLNET_DIRECTORY.
        tasks_wanted: List of tasks from dataset. Defaults to all available tasks.
        data_dir: Location to write CSV files. Defaults to /tmp/molnet/{name}/.

    Returns:
        tasks_wanted, (train_df, valid_df, test_df), transformers, out_paths

    """
    if data_dir is None:
        data_dir = os.path.join("/tmp/molnet/", name)
    os.makedirs(data_dir, exist_ok=True)

    tasks, dataframes, transformers = load_molnet_dataset(
        name, split=split, tasks_wanted=tasks_wanted, df_format="chemprop"
    )

    out_paths = []
    for split_name, df in zip(["train", "val", "test"], dataframes):
        path = os.path.join(data_dir, f"{split_name}.csv")
        out_paths.append(path)
        df.to_csv(path, index=False)

    return tasks, dataframes, transformers, out_paths


def make_dataframe(
    dataset, dataset_type, tasks, tasks_wanted, df_format: str = "chemberta"
):
    df = dataset.to_dataframe()
    if len(tasks) == 1:
        mapper = {"y": tasks[0]}
    else:
        mapper = {f"y{y_i+1}": task for y_i, task in enumerate(tasks_wanted)}
    df.rename(mapper, axis="columns", inplace=True)

    # Canonicalize SMILES
    smiles_list = [Chem.MolToSmiles(s, isomericSmiles=True) for s in df["X"]]

    # Convert labels to integer for classification
    labels = df[tasks_wanted]
    if dataset_type == "classification":
        labels = labels.astype(int)

    elif dataset_type == "regression":
        labels = labels.astype(float)

    if df_format == "chemberta":
        if len(tasks_wanted) == 1:
            labels = labels.values.flatten()
        else:
            # Convert labels to list for simpletransformers multi-label
            labels = labels.values.tolist()
        return pd.DataFrame({"text": smiles_list, "labels": labels})
    elif df_format == "chemprop":
        df_out = pd.DataFrame({"smiles": smiles_list})
        for task in tasks_wanted:
            df_out[task] = labels[task]
        return df_out
    else:
        raise ValueError(df_format)
