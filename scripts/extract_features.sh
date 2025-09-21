python3 extract_features.py \
    --dataset_path ../data/test.csv \
    --output ../data/seq_embeddings/test \
    --device mps

python3 extract_features.py \
    --dataset_path ../data/training.csv \
    --output ../data/seq_embeddings/train \
    --device mps