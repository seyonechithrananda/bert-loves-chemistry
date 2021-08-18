from absl import flags


def roberta_model_configuration_flags():
    flags.DEFINE_float(
        name="attention_probs_dropout_prob",
        default=0.1,
        help="The dropout ratio for the attention probabilities.",
        module_name="model",
    )
    flags.DEFINE_float(
        name="hidden_dropout_prob",
        default=0.1,
        help="The dropout probability for all fully connected layers in the embeddings, encoder, and pooler",
        module_name="model",
    )
    flags.DEFINE_integer(
        name="hidden_size_per_attention_head",
        default=64,
        help="Multiply with `num_attention_heads` to get the dimensionality of the encoder layers and the pooler layer.",
        module_name="model",
    )
    flags.DEFINE_integer(
        name="intermediate_size",
        default=3072,
        help="Dimensionality of the “intermediate” (often named feed-forward) layer in the Transformer encoder.",
        module_name="model",
    )
    flags.DEFINE_integer(
        name="max_position_embeddings",
        default=515,
        help="The maximum sequence length that this model might ever be used with. Typically set this to something large just in case (e.g., 512 or 1024 or 2048)",
        module_name="model",
    )
    flags.DEFINE_integer(
        name="num_attention_heads",
        default=6,
        help="Number of attention heads for each attention layer in the Transformer encoder.",
        module_name="model",
    )
    flags.DEFINE_integer(
        name="num_hidden_layers",
        default=6,
        help="Number of hidden layers in the Transformer encoder",
        module_name="model",
    )
    flags.DEFINE_integer(
        name="type_vocab_size",
        default=1,
        help="The vocabulary size of the token_type_ids passed when calling BertModel or TFBertModel",
        module_name="model",
    )
    flags.DEFINE_integer(
        name="vocab_size",
        default=600,
        help="Vocabulary size of the BERT model. Defines the number of different tokens that can be represented by the inputs_ids passed when calling BertModel or TFBertModel",
        module_name="model",
    )


def tokenizer_flags():
    flags.DEFINE_string(
        name="tokenizer_path",
        default="seyonec/SMILES_tokenized_PubChem_shard00_160k",
        help="Path to vocab file",
        module_name="tokenizer",
    )
    flags.DEFINE_integer(
        name="tokenizer_max_length",
        default=512,
        help="Controls the maximum length to use by one of the truncation/padding parameters for the tokenizer.",
        module_name="tokenizer",
    )


def dataset_flags():
    flags.DEFINE_string(name="dataset_path", default=None, help="Path to local dataset")
    flags.DEFINE_string(
        name="eval_path",
        default=None,
        help="If provided, uses dataset at this path as a validation set. Otherwise, `frac_train` is used to split the dataset.",
        module_name="dataset",
    )
    flags.DEFINE_float(
        name="frac_train",
        default=0.95,
        help="Fraction of dataset to use for training. Gets overridden by `eval_path`, if provided.",
        module_name="dataset",
    )
    flags.DEFINE_string(
        name="output_dir",
        default="default_dir",
        help="Directory in which to write results",
        module_name="dataset",
    )
    flags.DEFINE_string(
        name="run_name",
        default="default_run",
        help="Subdirectory for results",
        module_name="dataset",
    )


def train_flags():
    flags.DEFINE_integer(
        name="early_stopping_patience",
        default=3,
        help="Patience for the `EarlyStoppingCallback`.",
        module_name="training",
    )
    flags.DEFINE_integer(
        name="eval_steps",
        default=50,
        help="Number of update steps between two evaluations if evaluation_strategy='steps'. Will default to the same value as logging_steps if not set.",
        module_name="training",
    )
    flags.DEFINE_float(
        name="learning_rate",
        default=5e-5,
        help="The initial learning rate for AdamW optimizer",
        module_name="training",
    )
    flags.DEFINE_boolean(
        name="load_best_model_at_end",
        default=True,
        help="Whether or not to load the best model found during training at the end of training. When set to True (for HF 4.6), the parameters `save_strategy` and `save_steps` will be ignored and the model will be saved after each evaluation.",
        module_name="training",
    )
    flags.DEFINE_integer(
        name="logging_steps",
        default=10,
        help="Number of update steps between two logs if logging_strategy='steps'.",
        module_name="training",
    )
    flags.DEFINE_float(
        name="num_train_epochs",
        default=100,
        help="Total number of training epochs to perform (if not an integer, will perform the decimal part percents of the last epoch before stopping training).",
        module_name="training",
    )
    flags.DEFINE_boolean(
        name="overwrite_output_dir",
        default=True,
        help="If True, overwrite the content of the output directory. Use this to continue training if output_dir points to a checkpoint directory.",
        module_name="training",
    )
    flags.DEFINE_integer(
        name="per_device_train_batch_size",
        default=64,
        help="The batch size per GPU/TPU core/CPU for training.",
        module_name="training",
    )
    flags.DEFINE_integer(
        name="save_total_limit",
        default=None,
        help="If a value is passed, will limit the total amount of checkpoints. Deletes the older checkpoints in output_dir.",
        module_name="training",
    )
    flags.DEFINE_integer(
        name="save_steps",
        default=500,
        help="Number of updates steps before two checkpoint saves if `save_strategy`='steps'",
        module_name="training",
    )
    flags.DEFINE_float(
        name="mlm_probability",
        default=0.15,
        lower_bound=0.0,
        upper_bound=1.0,
        help="Masking rate",
        module_name="training",
    )
    flags.DEFINE_string(
        name="cloud_directory",
        default=None,
        help="If provided, syncs the run directory here using a callback.",
        module_name="training",
    )
