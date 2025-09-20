python3 extract_features.py \
    --dataset_path ../data/train.csv \
    --output_path ../data/seq_embeddings/train \
    --device cuda

python3 extract_features.py \
    --dataset_path ../data/test.csv \
    --output ../data/seq_embeddings/test \
    --device cuda