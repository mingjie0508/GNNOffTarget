#!/bin/bash
#SBATCH -p gpu
#SBATCH -c 8
#SBATCH --mem 64G
#SBATCH --gres gpu:1
#SBATCH -A kumargroup_gpu
#SBATCH -t 3-00:00
#SBATCH -J extract_features
#SBATCH -o /cluster/projects/kumargroup/hayden/slurm_logs/hackathon/extract_features_%j.out
#SBATCH -e /cluster/projects/kumargroup/hayden/slurm_logs/hackathon/extract_features_%j.err
#SBATCH --mail-type ALL
#SBATCH --mail-user cheukhei.yu@uhn.ca

###########################################################################################################

# Test
python3 extract_features.py \
    --dataset_path /cluster/projects/kumargroup/hayden/hackathon/data/test.csv \
    --output /cluster/projects/kumargroup/hayden/hackathon/data/seq_embeddings/test \
    --device cuda

# Train
python3 extract_features.py \
    --dataset_path /cluster/projects/kumargroup/hayden/hackathon/data/train.csv \
    --output /cluster/projects/kumargroup/hayden/hackathon/data/seq_embeddings/train \
    --device cuda