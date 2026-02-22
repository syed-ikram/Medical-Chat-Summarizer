import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import json
import argparse

class MedicalChatSummarizer:
    """Summarizer class for medical conversations"""
    
    def __init__(self, model_path):
        """
        Initialize the summarizer
        
        Args:
            model_path: Path to the fine-tuned model
        """
        print(f"Loading model from {model_path}...")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Load model
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_path,
            device_map="auto" if torch.cuda.is_available() else None,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        self.model.eval()
        
        # Load config if available
        try:
            with open(f"{model_path}/config.json", 'r') as f:
                self.config = json.load(f)
        except:
            self.config = {
                'max_source_length': 512,
                'max_target_length': 256
            }
        
        print("Model loaded successfully!")
        print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    
    def summarize(self, dialogue, max_length=None, num_beams=4, temperature=0.7):
        """
        Generate SOAP summary from dialogue
        
        Args:
            dialogue (str): Medical conversation
            max_length (int): Maximum summary length
            num_beams (int): Number of beams for generation
            temperature (float): Sampling temperature
        
        Returns:
            str: Generated SOAP summary
        """
        if max_length is None:
            max_length = self.config.get('max_target_length', 256)
        
        # Create prompt
        prompt = f"Summarize this medical conversation in SOAP format:\n\n{dialogue}\n\nSummary:"
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=self.config.get('max_source_length', 512),
            truncation=True,
            padding=True
        )
        
        # Move to device
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_beams=num_beams,
                early_stopping=True,
                no_repeat_ngram_size=3,
                temperature=temperature
            )
        
        # Decode
        summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Clean up
        if "Summary:" in summary:
            summary = summary.split("Summary:")[-1].strip()
        
        return summary
    
    def batch_summarize(self, dialogues, max_length=None, num_beams=4):
        """
        Generate summaries for multiple dialogues
        
        Args:
            dialogues (list): List of medical conversations
            max_length (int): Maximum summary length
            num_beams (int): Number of beams
        
        Returns:
            list: Generated summaries
        """
        summaries = []
        for dialogue in dialogues:
            summary = self.summarize(dialogue, max_length, num_beams)
            summaries.append(summary)
        return summaries

def main():
    """Main function for CLI usage"""
    parser = argparse.ArgumentParser(description="Medical Chat Summarization - Inference")
    parser.add_argument(
        '--model_path',
        type=str,
        default='./medical-chat-summarizer/final_model',
        help='Path to the fine-tuned model'
    )
    parser.add_argument(
        '--dialogue',
        type=str,
        default=None,
        help='Medical dialogue to summarize'
    )
    parser.add_argument(
        '--max_length',
        type=int,
        default=256,
        help='Maximum summary length'
    )
    parser.add_argument(
        '--num_beams',
        type=int,
        default=4,
        help='Number of beams for generation'
    )
    
    args = parser.parse_args()
    
    # Initialize summarizer
    summarizer = MedicalChatSummarizer(args.model_path)
    
    # Use provided dialogue or default example
    if args.dialogue:
        dialogue = args.dialogue
    else:
        dialogue = """
Doctor: Good morning. What brings you in today?
Patient: I've been having severe headaches for the past week.
Doctor: Can you describe the pain? Is it sharp or dull?
Patient: It's a throbbing pain, mostly on the right side of my head.
Doctor: Have you experienced any nausea or sensitivity to light?
Patient: Yes, both. The light really bothers me.
Doctor: Based on your symptoms, this appears to be a migraine. I'll prescribe some medication.
Patient: Thank you, doctor.
        """
    
    # Generate summary
    print("\n" + "=" * 80)
    print("MEDICAL DIALOGUE")
    print("=" * 80)
    print(dialogue)
    
    print("\n" + "=" * 80)
    print("GENERATED SOAP SUMMARY")
    print("=" * 80)
    
    summary = summarizer.summarize(
        dialogue,
        max_length=args.max_length,
        num_beams=args.num_beams
    )
    
    print(summary)
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
