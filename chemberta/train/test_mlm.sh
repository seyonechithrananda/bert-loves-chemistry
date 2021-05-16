python train_roberta.py \
    --model_type=mlm \
    --dataset_path=/home/ubuntu/src/bert-loves-chemistry/chemberta/data/pubchem_1k_smiles.txt\
    --model_name=test_mlm_1k \
    --per_device_train_batch_size=8 \
    --num_hidden_layers=2 \
    --num_attention_heads=2