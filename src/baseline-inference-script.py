import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from data import TrainDataset
from torch.utils.data import DataLoader
import argparse
import logging
from tqdm import tqdm
import wandb
import jiwer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def calculate_wer(predictions, ground_truths):
    return jiwer.wer(ground_truths, predictions)

def main(args):
    # Initialize wandb
    if args.use_wandb:
        wandb.init(project=args.wandb_project, entity=args.wandb_entity, name=args.wandb_run_name, mode=args.wandb_mode)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load pre-trained model and processor
    whisper_flavour = "/gpfs/projects/bsc88/speech/research/models/hf_models/whisper-tiny"
    processor = WhisperProcessor.from_pretrained(whisper_flavour)
    model = WhisperForConditionalGeneration.from_pretrained(whisper_flavour).to(device)
    print(whisper_flavour)
    # Create test dataset and dataloader
    test_dataset = TrainDataset(
        utterances_paths=args.test_data_path,
        processor=processor,
        random_crop_secs=0,  # No random cropping for inference
        tokens_max_length=args.tokens_max_length,
        speech_representation="mel",
        prompt_use_rate=0,  # No prompts for baseline
        prompt_length=0,
        vocab_size=51865,  # Whisper's vocab size
        nmels=80,
        padding_type="zero_pad",
        augmentation_prob=0,  # No augmentation for inference
        sample_rate=16000
    )

    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Inference
    model.eval()
    all_predictions = []
    all_ground_truths = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Inference"):
            input_features, labels = batch
            input_features = input_features.to(device)
            
            # Generate predictions
            forced_decoder_ids = processor.get_decoder_prompt_ids(language="ca", task="transcribe")
            predicted_ids = model.generate(
                input_features, 
                forced_decoder_ids=forced_decoder_ids,
                language="ca",  # Specify Catalan as the target language
                task="transcribe"
            )
            
            # Decode predictions and ground truths
            predictions = processor.batch_decode(predicted_ids, skip_special_tokens=True)
            ground_truths = processor.batch_decode(labels, skip_special_tokens=True)
            
            all_predictions.extend(predictions)
            all_ground_truths.extend(ground_truths)

    # Calculate WER
    wer = calculate_wer(all_predictions, all_ground_truths)
    logger.info(f"Word Error Rate: {wer}")

    if args.use_wandb:
        wandb.log({"baseline_wer": wer})
        
        # Log some examples
        num_examples = min(5, len(all_predictions))
        example_table = wandb.Table(columns=["Prediction", "Ground Truth"])
        for pred, truth in zip(all_predictions[:num_examples], all_ground_truths[:num_examples]):
            example_table.add_data(pred, truth)
        wandb.log({"baseline_examples": example_table})

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run baseline inference with Whisper model")
    parser.add_argument("--test_data_path", type=str, required=True, help="Path to the test dataset")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for inference")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of workers for data loading")
    parser.add_argument("--tokens_max_length", type=int, default=448, help="Maximum length of token sequences")
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases for logging")
    parser.add_argument("--wandb_project", type=str, default="SoftPrompting", help="Wandb project name")
    parser.add_argument("--wandb_entity", type=str, default="bsc", help="Wandb entity")
    parser.add_argument("--wandb_run_name", type=str, default="baseline-run-medium-devlt", help="Wandb run name")
    parser.add_argument("--wandb_mode", type=str, default="offline", help="Wandb mode, offline or online")
    
    args = parser.parse_args()
    main(args)