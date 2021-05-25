python train_roberta.py \
    --model_type=regression_lazy \
    --dataset_path=../data/pubchem_1k_smiles.txt \
    --normalization_path=../data/pubchem_descriptors_sample_1k_normalization_values_207.json \
    --output_dir=test_1k \
    --run_name=regression \
    --per_device_train_batch_size=8 \
    --num_hidden_layers=2 \
    --num_attention_heads=2