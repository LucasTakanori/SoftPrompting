import csv
import torch
import numpy as np
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from jiwer import wer
import librosa

def load_csv(file_path):
    data = []
    with open(file_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile, delimiter='\t')
        for row in reader:
            data.append(row)
    return data

def load_audio(file_path):
    audio, sr = librosa.load(file_path, sr=16000)
    return audio

def clean_text(input_text):
    ''' Remove special characters in 'remove_chars' list. '''
    remove_chars = ['.', ',', ';', ':', '¿', '?', '«', '»', '-', '¡', '!', '@', '*', '{', '}', '[', ']', '=', '/', '\\', '&', '#', '…']
    output_text = ''.join(char if char not in remove_chars else ' ' for char in input_text)
    return ' '.join(output_text.split()).lower()  # remove extra spaces and return cleaned text

def calculate_wer(reference, hypothesis):
    # Clean both reference and hypothesis before calculating WER
    clean_reference = clean_text(reference)
    clean_hypothesis = clean_text(hypothesis)
    return wer(clean_reference, clean_hypothesis)

def main(csv_file_path):
    # Load Whisper model and processor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny").to(device)
    processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")

    # Load CSV data
    data = load_csv(csv_file_path)

    total_wer = 0
    file_count = 0

    for item in data:
        audio_path = item['audio_path']
        reference_text = item['text']

        # Load and preprocess audio
        audio = load_audio(audio_path)
        input_features = processor(audio, sampling_rate=16000, return_tensors="pt").input_features.to(device)

        # Generate token ids
        forced_decoder_ids = processor.get_decoder_prompt_ids(language="ca", task="transcribe")
        
        # Generate logits
        with torch.no_grad():
            generated_ids = model.generate(
                input_features, 
                forced_decoder_ids=forced_decoder_ids,
                max_length=448
            )

        # Decode ids to text
        transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        # Calculate WER (this now uses the clean_text function internally)
        file_wer = calculate_wer(reference_text, transcription)
        total_wer += file_wer
        file_count += 1

        print(f"File: {audio_path}")
        print(f"Reference: {reference_text}")
        print(f"Cleaned Reference: {clean_text(reference_text)}")
        print(f"Transcription: {transcription}")
        print(f"Cleaned Transcription: {clean_text(transcription)}")
        print(f"WER: {file_wer}")
        print("---")

    # Calculate and print average WER
    average_wer = total_wer / file_count
    print(f"Average WER: {average_wer}")


if __name__ == "__main__":
    csv_file_path = "/home/usuaris/veu/lucas.takanori/lt400/dev.tsv"
    main(csv_file_path)