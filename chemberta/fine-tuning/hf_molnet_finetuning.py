# -*- coding: utf-8 -*-

"""
Work in progress.

#todo 
*   add FLAGs
*   clean up methods for using either trainer or pure-pytorch fine-tuning
*   Add weights and biases logging
*   add early stopping
*   align weight decay with class sizes
*   align warmup steps with simple-transformers setup
*   match training arguments to simple-transformers script


"""

import deepchem
from rdkit import Chem
import os
import numpy as np
import pandas as pd
from typing import List

import torch
from torch.utils.data import DataLoader


# import molnet loaders from deepchem
from deepchem.molnet import load_bbbp, load_clearance, load_clintox, load_delaney, load_hiv, load_qm7, load_tox21
from rdkit import Chem

# import MolNet dataloder from bert-loves-chemistry fork
from chemberta.utils.molnet_dataloader import load_molnet_dataset, write_molnet_dataset_for_chemprop
from chemberta.utils.molnet_dataset import MolNetDataset
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification, Trainer, TrainingArguments, AdamW




def read_molnet_df(df):
    texts = []
    labels = []

    for index, row in df.iterrows():
      texts.append(row['text'])
      labels.append(row['labels'])

    return texts, labels


def trainer_finetune():
    """#Fine-tuning using HF Trainer"""

    training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=3,              # total number of training epochs
    per_device_train_batch_size=16,  # batch size per device during training
    per_device_eval_batch_size=64,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=10,
    )

    model = RobertaForSequenceClassification.from_pretrained("seyonec/SMILES_tokenized_PubChem_shard00_160k")

    trainer = Trainer(
        model=model,                         # the instantiated 🤗 Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        train_dataset=train_dataset,         # training dataset
        eval_dataset=val_dataset             # evaluation dataset
    )

    trainer.train()

def torch_finetune():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = RobertaForSequenceClassification.from_pretrained("seyonec/SMILES_tokenized_PubChem_shard00_160k")
    model.to(device)
    model.train()

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    optim = AdamW(model.parameters(), lr=5e-5)

    for epoch in range(3):
        for batch in train_loader:
            optim.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs[0]
            loss.backward()
            optim.step()

    model.eval()


def main(argv):
    tasks, (train_df, valid_df, test_df), transformers = load_molnet_dataset("clintox", tasks_wanted=None)

    train_texts, train_labels = read_molnet_df(train_df)
    val_texts, val_labels = read_molnet_df(valid_df)
    test_texts, test_labels = read_molnet_df(test_df)

    tokenizer = RobertaTokenizerFast.from_pretrained('seyonec/SMILES_tokenized_PubChem_shard00_160k')

    train_encodings = tokenizer(train_texts, truncation=True, padding=True)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True)
    test_encodings = tokenizer(test_texts, truncation=True, padding=True)

    train_dataset = MolNetDataset(train_encodings, train_labels)
    val_dataset = MolNetDataset(val_encodings, val_labels)
    test_dataset = MolNetDataset(test_encodings, test_labels)

    
    trainer_finetune()
    torch_finetune


if __name__ == '__main__':
    app.run(main)



