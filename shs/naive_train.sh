#!/bin/bash
#SBATCH -o /home/usuaris/veussd/marc.casals/SoftPrompting/logs/outputs/slurm-%j.out
#SBATCH -e /home/usuaris/veussd/marc.casals/SoftPrompting/logs/errors/slurm-%j.err
#SBATCH -p veu             # Partition to submit to
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --gres=gpu:2
#SBATCH --job-name=naive-train
#SBATCH --mail-type=ALL
#SBATCH --mail-user=marc.casals@bsc.es

python src/train.py \
    --utterances_path "/home/usuaris/veussd/lucas.takanori/lt400/lt400.json",
    --seed 1234,
    