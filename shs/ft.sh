#!/bin/bash
#SBATCH -o /home/usuaris/veu/lucas.takanori/SoftPrompting/logs/ft/outputs/slurm-%j.out
#SBATCH -e /home/usuaris/veu/lucas.takanori/SoftPrompting/logs/ft/errors/slurm-%j.err
#SBATCH -p veu             # Partition to submit to
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=ft_whisper
##SBATCH --mail-type=ALL
##SBATCH --mail-user=marc.casals@bsc.es

export WANDB_API_KEY='805aa556027c20132de063b7c79fb877840d510d'

python src/whisper_finetuning.py --use_wandb --output_dir ./whisper_ft_ca2
    