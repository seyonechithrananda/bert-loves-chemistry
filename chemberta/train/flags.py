from absl import flags


def roberta_model_configuration():
    # RobertaConfig params
    flags.DEFINE_integer(
        name="vocab_size",
        default=600,
        help="Vocabulary size of the BERT model. Defines the number of different tokens that can be represented by the inputs_ids passed when calling BertModel or TFBertModel",
    )
    flags.DEFINE_integer(
        name="intermediate_size",
        default=3072,
        help="Dimensionality of the “intermediate” (often named feed-forward) layer in the Transformer encoder.",
    )
    flags.DEFINE_integer(
        name="max_position_embeddings",
        default=515,
        help="The maximum sequence length that this model might ever be used with. Typically set this to something large just in case (e.g., 512 or 1024 or 2048)",
    )
    flags.DEFINE_integer(
        name="num_attention_heads",
        default=6,
        help="Number of attention heads for each attention layer in the Transformer encoder.",
    )
    flags.DEFINE_integer(
        name="num_hidden_layers",
        default=6,
        help="Number of hidden layers in the Transformer encoder",
    )
    flags.DEFINE_integer(
        name="type_vocab_size",
        default=1,
        help="The vocabulary size of the token_type_ids passed when calling BertModel or TFBertModel",
    )
    flags.DEFINE_float(
        name="hidden_dropout_prob",
        default=0.1,
        help="The dropout probability for all fully connected layers in the embeddings, encoder, and pooler",
    )
    flags.DEFINE_float(
        name="attention_probs_dropout_prob",
        default=0.1,
        help="The dropout ratio for the attention probabilities.",
    )
