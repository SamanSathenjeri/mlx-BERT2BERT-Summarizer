import math
import yaml
import numpy as np
from dataclasses import dataclass
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import mlx.utils as utils
from tqdm import tqdm

from datasets import load_dataset, concatenate_datasets
from transformers import BertTokenizerFast
from model import BERT, BERT_Config

def train(model, train_data, batch_size, num_epochs, learning_rate):
    optimizer = optim.Adam(model.parameters(), learning_rate=learning_rate)
    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = len(train_data['input_ids']) // batch_size

        for i in tqdm(range(num_batches), desc=f"Epoch {epoch+1}"):
            start = i * batch_size
            end = start + batch_size
            batch_input = mx.array(train_data['input_ids'][start:end], dtype=mx.int32)
            batch_labels = mx.array(train_data['labels'][start:end], dtype=mx.int32)

            def loss_fn():
                logits = model(batch_input)
                loss = nn.losses.cross_entropy(logits, batch_labels, ignore_index=-100)
                return loss

            loss, grads = mx.value_and_grad(loss_fn)()
            optimizer.update(model.parameters(), grads)

            total_loss += loss.item()

        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1} avg loss: {avg_loss:.4f}")

if __name__ == "__main__":
    config = BERT_Config.from_yaml()
    model = BERT(config)

    print("model setup")

    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    block_size = 128

    print("loaded dataset")

    def tokenize_function(examples):
        return tokenizer(examples["text"], return_special_tokens_mask=True)

    tokenized = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    concatenated = concatenate_datasets([tokenized, tokenized]) 
    input_ids = sum(concatenated["input_ids"], [])
    input_ids = [i for i in input_ids if i != tokenizer.pad_token_id]

    print("starting chunking")

    def chunkify(lst, size):
        return [lst[i:i+size] for i in range(0, len(lst)-size, size)]

    chunks = chunkify(input_ids, block_size)
    chunks = np.array(chunks, dtype=np.int32)

    print("creating masking labels")

    def mask_tokens(inputs, tokenizer, mlm_probability=0.15):
        labels = np.full_like(inputs, -100)

        probability_matrix = np.random.rand(*inputs.shape)
        mask_arr = probability_matrix < mlm_probability
        labels[mask_arr] = inputs[mask_arr]

        rand = np.random.rand(*inputs.shape)
        mask_token_id = tokenizer.mask_token_id

        mask_replace = mask_arr & (rand < 0.8)
        random_replace = mask_arr & (rand >= 0.8) & (rand < 0.9)
        unchanged = mask_arr & (rand >= 0.9)

        inputs[mask_replace] = mask_token_id
        inputs[random_replace] = np.random.randint(0, tokenizer.vocab_size, random_replace.sum())

        return inputs, labels

    input_ids_masked, labels = mask_tokens(chunks.copy(), tokenizer)

    train_data = {
        "input_ids": input_ids_masked,
        "labels": labels
    }

    print("starting to train")

    train(model, train_data, batch_size=16, num_epochs=3, learning_rate=1e-4)