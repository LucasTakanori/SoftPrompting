import torch
from torch.utils.data import Dataset, DataLoader
from transformers import WhisperProcessor, WhisperForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer
import pandas as pd
import numpy as np
import librosa
import argparse
import logging
from tqdm import tqdm
import wandb
from datasets import load_metric

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WhisperTSVDataset(Dataset):
    def __init__(self, tsv_path, processor, max_length=444):
        self.df = pd.read_csv(tsv_path, sep='\t')
        self.processor = processor
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        audio_path = self.df.iloc[idx]['audio_path']
        text = self.df.iloc[idx]['text']

        # Load audio
        speech, sr = librosa.load(audio_path, sr=16000)

        # Compute log-mel spectrogram
        input_features = self.processor(speech, sampling_rate=16000, return_tensors="pt").input_features

        # Tokenize text
        labels = self.processor.tokenizer(text, max_length=self.max_length, truncation=True).input_ids

        return input_features.squeeze(), labels

class WhisperDataCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, features):
        # Extract input_features and labels
        input_features = [{"input_features": feature[0]} for feature in features]
        labels = [feature[1] for feature in features]

        # Pad input features
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # Pad and process labels
        label_features = self.processor.tokenizer.pad({"input_ids": labels}, return_tensors="pt")
        
        # Replace padding with -100 to ignore loss on padding tokens
        label_features['input_ids'][label_features['input_ids'] == self.processor.tokenizer.pad_token_id] = -100

        # Add labels to batch
        batch["labels"] = label_features['input_ids']

        return batch

def calculate_wer(pred):
    wer_metric = load_metric("wer")
    predictions = pred.predictions
    labels = pred.label_ids

    # Decode predictions and labels
    pred_str = processor.batch_decode(predictions, skip_special_tokens=True)
    label_str = processor.batch_decode(labels, skip_special_tokens=True)

    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    
    # Check if wer is a float (single value) or a dict
    if isinstance(wer, float):
        return {"wer": wer}
    else:
        return wer

class CatalanWhisperTrainer(Seq2SeqTrainer):
    def __init__(self, *args, max_new_tokens=448, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_new_tokens = max_new_tokens

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            if self.args.predict_with_generate:
                generation_inputs = {
                    "input_features": inputs["input_features"],
                    "attention_mask": inputs.get("attention_mask", None),
                    "forced_decoder_ids": processor.get_decoder_prompt_ids(language="ca", task="transcribe"),
                    "language": "ca",
                    "task": "transcribe",
                }
                generated_tokens = model.generate(**generation_inputs, max_new_tokens=self.max_new_tokens)
                loss = None
            else:
                outputs = model(**inputs)
                loss = outputs.loss
                generated_tokens = outputs.logits.argmax(dim=-1)

        labels = inputs["labels"]
        return loss, generated_tokens, labels

def main(args):
    # Initialize wandb
    if args.use_wandb:
        wandb.init(project=args.wandb_project, entity=args.wandb_entity, name=args.wandb_run_name)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load pre-trained model and processor
    global processor
    processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny").to(device)

    # Create train and validation datasets
    train_dataset = WhisperTSVDataset(args.train_data_path, processor, max_length=args.tokens_max_length)
    val_dataset = WhisperTSVDataset(args.val_data_path, processor, max_length=args.tokens_max_length)

    # Define training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        weight_decay=args.weight_decay,
        num_train_epochs=args.num_epochs,
        predict_with_generate=True,
        fp16=args.fp16,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        push_to_hub=False,
        report_to="wandb" if args.use_wandb else "none",
    )

    # Create data collator
    data_collator = WhisperDataCollator(processor)

    # Define trainer
    trainer = CatalanWhisperTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=calculate_wer,
        max_new_tokens=args.tokens_max_length,
    )

    # Fine-tune the model
    trainer.train()

    # Save the fine-tuned model
    trainer.save_model(args.output_dir)

    # Evaluate the model on the validation set
    eval_results = trainer.evaluate()
    logger.info(f"Evaluation results: {eval_results}")

    if args.use_wandb:
        wandb.log({"final_wer": eval_results["eval_wer"]})

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune Whisper model for Catalan")
    parser.add_argument("--train_data_path", type=str, default="/home/usuaris/veu/lucas.takanori/lt400/train.tsv", help="Path to the training dataset TSV file")
    parser.add_argument("--val_data_path", type=str, default="/home/usuaris/veu/lucas.takanori/lt400/test.tsv", help="Path to the validation dataset TSV file")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the fine-tuned model")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training and evaluation")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--tokens_max_length", type=int, default=444, help="Maximum length of token sequences")
    parser.add_argument("--eval_steps", type=int, default=500, help="Number of steps between evaluations")
    parser.add_argument("--logging_steps", type=int, default=100, help="Number of steps between logging")
    parser.add_argument("--save_steps", type=int, default=1000, help="Number of steps between saving checkpoints")
    parser.add_argument("--fp16", action="store_true", help="Use mixed precision training")
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases for logging")
    parser.add_argument("--wandb_project", type=str, default="SoftPrompting", help="Wandb project name")
    parser.add_argument("--wandb_entity", type=str, default="bsc", help="Wandb entity")
    parser.add_argument("--wandb_run_name", type=str, default="whisper-finetuning-catalan", help="Wandb run name")
    
    args = parser.parse_args()
    main(args)