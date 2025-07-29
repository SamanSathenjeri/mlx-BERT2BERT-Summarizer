import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from tqdm import tqdm

from datasets import load_dataset, concatenate_datasets
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from model import BERT, BERT_Config

if __name__ == "__main__":
    model_path = './Finetuned_BERT_weights.safetensors'

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, use_safetensors=True)

    while True:
        user_input = input("> ") # Get input with the "> " prefix
        
        if user_input.lower() == 'exit': # Allow a way to break out of the loop
            print("Exiting the program.")
            break

        inputs = tokenizer(user_input, return_tensors="pt")
