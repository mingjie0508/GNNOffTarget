python3 extract_features.py \
    --dataset_path ../data/train.csv \
    --output ../data/seq_embeddings/train \
    --device mps

python3 extract_features.py \
    --dataset_path ../data/test.csv \
    --output ../data/seq_embeddings/test \
    --device mps

    