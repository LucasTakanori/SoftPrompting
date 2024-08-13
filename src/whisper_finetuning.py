import torch
from torch.utils.data import Dataset
from transformers import WhisperProcessor, WhisperForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer
import pandas as pd
import librosa
import argparse
import logging
import wandb
import evaluate
import uuid
import os
from tqdm import tqdm

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
        speech, _ = librosa.load(audio_path, sr=16000)
        input_features = self.processor(speech, sampling_rate=16000, return_tensors="pt").input_features
        labels = self.processor.tokenizer(text, max_length=self.max_length, truncation=True).input_ids
        return input_features.squeeze(), labels

class WhisperDataCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, features):
        input_features = [{"input_features": feature[0]} for feature in features]
        labels = [feature[1] for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        label_features = self.processor.tokenizer.pad({"input_ids": labels}, return_tensors="pt")
        label_features['input_ids'][label_features['input_ids'] == self.processor.tokenizer.pad_token_id] = -100
        batch["labels"] = label_features['input_ids']
        return batch

def calculate_wer(pred):
    wer_metric = evaluate.load("wer")
    predictions = pred.predictions
    labels = pred.label_ids
    pred_str = processor.batch_decode(predictions, skip_special_tokens=True)
    label_str = processor.batch_decode(labels, skip_special_tokens=True)
    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}

class CatalanWhisperTrainer(Seq2SeqTrainer):
    def __init__(self, processor, *args, max_new_tokens=448, **kwargs):
        super().__init__(*args, **kwargs)
        self.processor = processor
        self.max_new_tokens = max_new_tokens
        self.step = 0

    def log(self, logs):
        logs = logs or {}
        self.step += 1
        
        if self.args.report_to == ["wandb"]:
            current_lr = self.optimizer.param_groups[0]['lr'] if self.optimizer else self.args.learning_rate
            wandb_logs = {
                "train/step": self.step,
                "train/loss": logs.get("loss", 0),
                "train/learning_rate": current_lr,
                "train/epoch": logs.get("epoch", 0)
            }
            logger.info(f"Logging to wandb: {wandb_logs}")
            wandb.log(wandb_logs)
        else:
            logger.info("Wandb logging is not enabled.")
        
        return super().log(logs)

    def calculate_wer(self, dataset):
        self.model.eval()
        all_predictions = []
        all_references = []
        
        for batch in tqdm(self.get_eval_dataloader(dataset), desc="Calculating WER"):
            with torch.no_grad():
                outputs = self.model.generate(
                    input_features=batch["input_features"].to(self.args.device),
                    forced_decoder_ids=self.processor.get_decoder_prompt_ids(language="ca", task="transcribe"),
                    max_new_tokens=self.max_new_tokens
                )
            predictions = self.processor.batch_decode(outputs, skip_special_tokens=True)
            references = self.processor.batch_decode(batch["labels"], skip_special_tokens=True)
            
            all_predictions.extend(predictions)
            all_references.extend(references)
        
        wer_metric = evaluate.load("wer")
        wer = wer_metric.compute(predictions=all_predictions, references=all_references)
        
        return wer

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix: str = "eval"):
        output = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        
        # Calculate WER for validation set
        val_wer = self.calculate_wer(eval_dataset or self.eval_dataset)
        
        # Calculate WER for training set
        train_wer = self.calculate_wer(self.train_dataset)
        
        logger.info(f"Validation WER: {val_wer:.4f}")
        logger.info(f"Training WER: {train_wer:.4f}")
        
        if self.args.report_to == ["wandb"]:
            wandb_logs = {
                "eval/val_wer": val_wer,
                "eval/train_wer": train_wer,
                "eval/loss": output.get("eval_loss", 0),
                "eval/runtime": output.get("eval_runtime", 0),
                "eval/samples_per_second": output.get("eval_samples_per_second", 0),
                "eval/steps_per_second": output.get("eval_steps_per_second", 0),
                "eval/epoch": output.get("epoch", 0),
                "eval/step": self.step
            }
            logger.info(f"Logging evaluation to wandb: {wandb_logs}")
            wandb.log(wandb_logs)
        
        return output

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            if self.args.predict_with_generate:
                generated_tokens = model.generate(
                    inputs["input_features"],
                    forced_decoder_ids=self.processor.get_decoder_prompt_ids(language="ca", task="transcribe"),
                    max_new_tokens=self.max_new_tokens
                )
                loss = None
            else:
                outputs = model(**inputs)
                loss = outputs.loss
                generated_tokens = outputs.logits.argmax(dim=-1)

        labels = inputs["labels"]
        return loss, generated_tokens, labels

def main(args):
    if args.use_wandb:
        run_id = str(uuid.uuid4())
        run_name = f"whisper-finetuning-{run_id[:8]}"
        wandb.init(
            project="SoftPrompting",
            entity="bsc",
            name=run_name,
            id=run_id,
            config=vars(args)
        )
        logger.info(f"Wandb initialized. Run name: {run_name}, Run ID: {run_id}")
        wandb.log({"test": 1})
        logger.info("Test log sent to wandb.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    global processor
    processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny").to(device)
    model.resize_token_embeddings(len(processor.tokenizer))

    train_dataset = WhisperTSVDataset(args.train_data_path, processor, max_length=args.tokens_max_length)
    val_dataset = WhisperTSVDataset(args.val_data_path, processor, max_length=args.tokens_max_length)

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

    trainer = CatalanWhisperTrainer(
        processor=processor,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=WhisperDataCollator(processor),
        compute_metrics=calculate_wer,
        max_new_tokens=args.tokens_max_length,
    )

    try:
        trainer.train()
        trainer.save_model(args.output_dir)
        
        eval_results = trainer.evaluate()
        logger.info(f"Final evaluation results: {eval_results}")

        if args.use_wandb:
            final_logs = {}
            for key, value in eval_results.items():
                final_logs[f"final_{key}"] = value
            
            logger.info(f"Logging final results to wandb: {final_logs}")
            wandb.log(final_logs)

            # Save the model as an artifact
            artifact = wandb.Artifact(f"model_{run_id}", type="model")
            artifact.add_dir(args.output_dir)
            wandb.log_artifact(artifact)
            
    except Exception as e:
        logger.error(f"An error occurred during training: {str(e)}")
    finally:
        if args.use_wandb:
            wandb.finish()
            logger.info("Wandb logging finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune Whisper model for Catalan")
    parser.add_argument("--train_data_path", type=str, default="/home/usuaris/veu/lucas.takanori/lt400/train.tsv", help="Path to the training dataset TSV file")
    parser.add_argument("--val_data_path", type=str, default="/home/usuaris/veu/lucas.takanori/lt400/test.tsv", help="Path to the validation dataset TSV file")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the fine-tuned model")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for training and evaluation")
    parser.add_argument("--num_epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--tokens_max_length", type=int, default=444, help="Maximum length of token sequences")
    parser.add_argument("--eval_steps", type=int, default=300, help="Number of steps between evaluations")
    parser.add_argument("--logging_steps", type=int, default=10, help="Number of steps between logging")
    parser.add_argument("--save_steps", type=int, default=900, help="Number of steps between saving checkpoints")
    parser.add_argument("--fp16", action="store_true", help="Use mixed precision training")
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases for logging")
    
    args = parser.parse_args()
    main(args)