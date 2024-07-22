import torch
from whisper.tokenizer import get_tokenizer

class WhisperDecoder:
    def __init__(self, model_flavor):
        self.tokenizer = get_tokenizer(model_flavor)
    
    def logits_to_words(self, logits):
        # Apply softmax to get probabilities
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        
        # Select the token with the highest probability for each position
        predicted_tokens = torch.argmax(probabilities, dim=-1)
        
        # Convert tokens to words
        predicted_tokens = predicted_tokens.squeeze().tolist()
        predicted_words = self.tokenizer.decode(predicted_tokens)
        
        return predicted_words

# Example usage
#logits = torch.randn(1, 448, 51865)  # Example logits tensor
logits = torch.zeros(1, 448, 51865)
logits[0, 1, 12345] = 0.5
print(logits)
decoder = WhisperDecoder('base')  # Replace 'your-whisper-flavor' with the actual model flavor
predicted_words = decoder.logits_to_words(logits)

print(predicted_words)
