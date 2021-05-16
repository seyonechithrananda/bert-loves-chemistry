python train_roberta.py \
    --model_type=regression \
    --dataset_path=/home/ubuntu/src/bert-loves-chemistry/chemberta/data/pubchem_descriptors_sample_1k_clean.csv \
    --normalization_path=/home/ubuntu/src/bert-loves-chemistry/chemberta/data/pubchem_descriptors_sample_1k_normalization_values.json \
    --model_name=test_mtr_1k \
    --per_device_train_batch_size=8 \
    --num_hidden_layers=2 \
    --num_attention_heads=2