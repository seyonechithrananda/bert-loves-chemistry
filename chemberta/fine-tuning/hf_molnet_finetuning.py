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
from absl import app
from absl import flags

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


FLAGS = flags.FLAGS

flags.DEFINE_string(name="molnet_dataset", defualt="clintox", help="")
flags.mark_flag_as_required('molnet_dataset')

flags.DEFINE_string(name="model_path", defualt="seyonec/SMILES_tokenized_PubChem_shard00_160k", help="")
flags.mark_flag_as_required('model_path')

flags.DEFINE_integer(name="num_train_epochs", default=10, help="")
flags.DEFINE_integer(name="per_device_train_batch_size", default=32, help="")
flags.DEFINE_integer(name="per_device_eval_batch_size", default=64, help="")
flags.DEFINE_integer(name="warmup_steps", default=500, help="")
flags.DEFINE_integer(name="weight_decay", default=0.01, help="")

flags.DEFINE_string(name='logging_dir', default='./logs', help="")
flags.DEFINE_integer(name='logging_steps', default=10, help="")
flags.DEFINE_string(name='output_dir', default='./results', help="")
flags.DEFINE_boolean(name='freeze_weights', default=False, help="")


def read_molnet_df(df):
    texts = []
    labels = []

    for index, row in df.iterrows():
      texts.append(row['text'])
      labels.append(row['labels'])

    return texts, labels


def trainer_finetune():
    #Fine-tuning using HF Trainer

    training_args = TrainingArguments(
    output_dir=FLAGS.output_dir,          # output directory
    num_train_epochs=FLAGS.num_train_epochs,          # total number of training epochs
    per_device_train_batch_size=FLAGS.per_device_train_batch_size,  # batch size per device during training
    per_device_eval_batch_size=FLAGS.per_device_eval_batch_size,   # batch size for evaluation
    warmup_steps=FLAGS.warmup_steps,          # number of warmup steps for learning rate scheduler
    weight_decay=FLAGS.weight_decay,          # strength of L2 weight decay (taken directly from roberta paper settings)
    logging_dir=FLAGS.logging_dir,          # directory for storing logs
    logging_steps=FLAGS.logging_steps,
    )

    model = RobertaForSequenceClassification.from_pretrained(FLAGS.model_path)

    # freeze weights and only train classifier head
    if FLAGS.freeze_weights:
        for param in model.roberta.parameters():
            param.requires_grad = False

    trainer = Trainer(
        model=model,                         # the instantiated 🤗 Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        train_dataset=train_dataset,         # training dataset
        eval_dataset=val_dataset             # evaluation dataset
    )

    trainer.train()

"""
def torch_finetune():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = RobertaForSequenceClassification.from_pretrained(FLAGS.model_path)
    model.to(device)
    model.train()

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    optim = AdamW(model.parameters(), lr=5e-5)

    for epoch in range(10):
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

"""

def main(argv):
    tasks, (train_df, valid_df, test_df), transformers = load_molnet_dataset(FLAGS.molnet_dataset, tasks_wanted=None)

    train_texts, train_labels = read_molnet_df(train_df)
    val_texts, val_labels = read_molnet_df(valid_df)
    test_texts, test_labels = read_molnet_df(test_df)

    tokenizer = RobertaTokenizerFast.from_pretrained(FLAGS.model_path)

    train_encodings = tokenizer(train_texts, truncation=True, padding=True)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True)
    test_encodings = tokenizer(test_texts, truncation=True, padding=True)

    train_dataset = MolNetDataset(train_encodings, train_labels)
    val_dataset = MolNetDataset(val_encodings, val_labels)
    test_dataset = MolNetDataset(test_encodings, test_labels)

    
    trainer_finetune()
    # torch_finetune()


if __name__ == '__main__':
    app.run(main)



