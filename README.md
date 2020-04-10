# bert-loves-chemistry
bert-loves-chemistry: a repository of HuggingFace models applied on chemical SMILES data for drug design, chemical modelling, etc.

Right now the notebooks are all for the RoBERTa model (a variant of BERT) trained on the task of masked-language modelling (MLM). Training was done over 5 epochs until loss converged to around 0.39. The model weights for training are available using [HuggingFace](https://huggingface.co/seyonec/ChemBERTa-zinc-base-v1). I hope this is of use to developers, students and researchers exploring the use of transformers and the attention mechanism for chemistry!

You can load the tokenizer + model for MLM prediction tasks using the following code:

```
from transformers import AutoModelWithLMHead, AutoTokenizer, pipeline

model = AutoModelWithLMHead.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
tokenizer = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")

fill_mask = pipeline('fill-mask', model=model, tokenizer=tokenizer)
```

Todo:
- [x]  Finish writing notebook to train model
- [x]  Finish notebook to preload and run predictions on a single molecule —> test if HuggingFace works
- [x]  Train RoBERTa model until convergence
- [x]  Upload weights onto HuggingFace
- [x]  Create tutorial using evaluation + fine-tuning notebook.
- [ ]  Create documentation + writing, visualizations for notebook.
- [ ]  Setup PR into DeepChem
