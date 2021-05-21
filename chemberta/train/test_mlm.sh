python train_roberta.py \
    --model_type=mlm \
    --dataset_path=../data/pubchem_1k_smiles.txt \
    --output_dir=test_1k \
    --run_name=mlm \
    --per_device_train_batch_size=8 \
    --num_hidden_layers=2 \
    --num_attention_heads=2