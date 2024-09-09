#!/bin/bash
#SBATCH --job-name=SoftpromptTinyDecoder
#SBATCH -D .
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1 
#SBATCH --cpus-per-task=80
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:4
#SBATCH --account bsc88
##SBATCH --exclusive
##SBATCH --qos acc_debug
#SBATCH -q acc_bscls          # QoS for life sciences in nodes with GPUs (acc_bscls)
#SBATCH --mail-type=ALL
#SBATCH --mail-user=lucas.sanchez@bsc.es
date
module load impi intel mkl hdf5 
module load EB/apps EB/install 
module load FFmpeg/6.0-GCCcore-13.2.0
#module load intel hdf5 mkl 
#export LD_LIBRARY_PATH=/gpfs/apps/MN5/ACC/HDF5/SRC/hdf5-hdf5-1_14_1-2/src/.libs/libhdf5.so.310:$LD_LIBRARY_PATH

source /gpfs/projects/bsc88/speech/research/environments/SoftPrompting/bin/activate

export WANDB_API_KEY='805aa556027c20132de063b7c79fb877840d510d'

export GPUS_PER_NODE=4


export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

export HF_HOME=/gpfs/projects/bsc88/speech/research/repos/SoftPrompting/CACHE
export HF_DATASETS_CACHE=/gpfs/projects/bsc88/speech/research/repos/SoftPrompting/CACHE
export CACHE_DIR=/gpfs/projects/bsc88/speech/research/repos/SoftPrompting/CACHE
#export HF_DATASET="/gpfs/projects/bsc88/speech/data/raw_data/1_DATALOADERS/CATALAN_DATALOADER/loading_script_whisper.py"

python src/train.py 

date
    