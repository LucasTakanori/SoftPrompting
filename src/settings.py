TRAIN_DEFAULT_SETTINGS = {
    "utterances_path": "/home/usuaris/veussd/lucas.takanori/lt400/lt400.json",
    "random_seed": 1234,
    "max_epochs": 1,
    "use_weights_and_biases": False,
    "load_checkpoint" : False,
    "eval_and_save_best_model_every": 100,
    "training_augmentation_prob": 0,
    "evaluation_augmentation_prob": 0,
    "sample_rate": 16000,
    "whisper_flavour": "tiny",
    "batch_size": 4,
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
    "optimizer": "adam"
}