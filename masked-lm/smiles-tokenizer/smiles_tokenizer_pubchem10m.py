# -*- coding: utf-8 -*-
"""
Colab file is located at
    https://colab.research.google.com/drive/15G0iwuxsr4xrMFSnjXxfmU94utPn1Zvm

Requirements:
- transformers
- wandb
- nlp
- torch
"""


import transformers
import torch
import wandb
from transformers import (
    RobertaConfig,
    RobertaTokenizerFast,
    RobertaForMaskedLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from nlp import load_dataset
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
        self.tokenizer = tokenizer
        self.file_path = file_path
        self.block_size = block_size

        self.dataset = load_dataset("text", data_files=file_path)["train"]
        print("Loaded Dataset")
        self.len = len(self.dataset)
        print("Number of lines: " + str(self.len))
        print("Block size: " + str(self.block_size))

    def __len__(self):
        return self.len

    def preprocess(self, text):
        batch_encoding = self.tokenizer(str(text), add_special_tokens=True, truncation=True, max_length=self.block_size)
        return torch.tensor(batch_encoding["input_ids"])

    def __getitem__(self, i):
        line = self.dataset[i]
        example = self.preprocess(line)
        return example


def main():
    # Main training script

    wandb.login()

    #verify file length
    fname = 'pubchem-10m.txt'
    def file_len(fname):
        with open(fname) as f:
            for i, l in enumerate(f):
                pass
        return i + 1

    print(file_len(fname))

    torch.cuda.is_available() #checking if CUDA + Colab GPU works

    config = RobertaConfig(
        vocab_size=52_000,
        max_position_embeddings=512,
        num_attention_heads=12,
        num_hidden_layers=6,
        type_vocab_size=1,
    )

    tokenizer = RobertaTokenizerFast.from_pretrained("seyonec/SMILES_tokenized_PubChem_shard00_160k", max_len=512)

    # test
    tokenizer.encode("[O-][N+](=O)c1cnc(s1)Sc1nnc(s1)N")

    model = RobertaForMaskedLM(config=config)
    model.num_parameters()

    dataset = RawTextDataset(tokenizer=tokenizer, file_path="pubchem-10m.txt", block_size=512)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )

    training_args = TrainingArguments(
        output_dir="PubChem_10M_SMILES_Tokenizer",
        overwrite_output_dir=True,
        num_train_epochs=10,
        per_gpu_train_batch_size=64,
        save_steps=10_000,
        save_total_limit=2,
        fp16 = True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
    )

    trainer.train()
    trainer.save_model("PubChem_10M_SMILES_Tokenizer")


if __name__ == "__main__":
    main()

"""# Methods for preventing RuntimeError

- reduce per_gpu_train_batch_size to 16/32 (longer training time)
- reduce seq_length to 128 or below (recommended to be 128 minimum)
"""

