#!/bin/bash
#SBATCH -o /home/usuaris/veussd/marc.casals/SoftPrompting/logs/outputs/slurm-%j.out
#SBATCH -e /home/usuaris/veussd/marc.casals/SoftPrompting/logs/errors/slurm-%j.err
#SBATCH -p veu             # Partition to submit to
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --gres=gpu:2
#SBATCH --job-name=peft-train
#SBATCH --mail-type=ALL
#SBATCH --mail-user=marc.casals@bsc.es


python TS-ASR/training_pt.py \
        --train-json \
        --embed_path \
        --batch-size 32 \
        --dev-batch-size 32 \
        --no-timestamps-training \
        --model tiny.en \
        --prompt_length \
        --lr \
        --exp_name \
        --seed \
