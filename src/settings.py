TRAIN_DEFAULT_SETTINGS = {
    "utterances_path": "/home/usuaris/veussd/lucas.takanori/lt400/train.tsv",
    "random_seed": 1234,
    "max_epochs": 5,
    "use_weights_and_biases": False,
    "load_checkpoint" : False,
    "checkpoint_file_folder": "./checkpoints",
    "checkpoint_file_name": "latest_checkpoint.pth",
    "training_augmentation_prob": 0,
    "evaluation_augmentation_prob": 0,
    "eval_batch_size" : 1,
    "sample_rate": 16000,
    "whisper_flavour": "tiny",
    "batch_size": 10,
    "num_workers": 2,
    "random_crop_secs": 30,
    "padding_type": "zero_pad",
    "asr_model": "whisper",
    "learning_rate": 1e-4,
    "tokens_max_length": 444,
    "prompt_depth": 32,
    "prompt_length": 100,
    "prompt_dim": 512,
    "speech_representation": "mel",
    "nmels": 80,
    "context_len": 100,
    "loss": "CrossEntropy",
    "optimizer": "adam",
    "vocab_size":51865 ,
    "validation_utterances_path":"/home/usuaris/veussd/lucas.takanori/lt400/dev.tsv",
    "eval_and_save_best_model_every": 30,
}