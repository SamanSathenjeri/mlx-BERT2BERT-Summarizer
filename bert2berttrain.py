import os
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer
from BERTEncoder import BERT_Config, BERT
from BERTDecoder import BERT2BERT

BATCH_SIZE = 8
SEQ_LEN = 256
NUM_EPOCHS = 10
LR = 1e-4
PAD_TOKEN_ID = 0
PRETRAINED_ENCODER_PATH = "safetensors/BERT_weights.safetensors"

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", model_max_length=SEQ_LEN)
tokenizer.pad_token = tokenizer.eos_token or tokenizer.sep_token or "[PAD]"

def preprocess(example):
    input_text = example["article"]
    target_text = example["highlights"]

    if not input_text.strip() or not target_text.strip():
        return None

    encoder = tokenizer(input_text, truncation=True, padding="max_length", max_length=SEQ_LEN, return_tensors="np")
    decoder = tokenizer(target_text, truncation=True, padding="max_length", max_length=SEQ_LEN, return_tensors="np")

    return {
        "encoder_input_ids": encoder["input_ids"][0],
        "decoder_input_ids": decoder["input_ids"][0][:-1],
        "labels": decoder["input_ids"][0][1:],
        "padding_mask": decoder["attention_mask"][0][1:].astype("float32"),
    }

raw_train_dataset = load_dataset("cnn_dailymail", "3.0.0", split="train", streaming=True).with_format("python")
val_raw_dataset = load_dataset("cnn_dailymail", "3.0.0", split="validation", streaming=True).with_format("python")

train_dataset = []
for e in raw_train_dataset:
    item = preprocess(e)
    if item is not None:
        train_dataset.append(item)

val_dataset = []
for e in val_raw_dataset:
    item = preprocess(e)
    if item is not None:
        val_dataset.append(item)

def create_batches(data, batch_size):
    filtered_data = [d for d in data if d is not None]
    if not filtered_data:
        return []
    for i in range(0, len(filtered_data), batch_size):
        batch = filtered_data[i:i+batch_size]
        yield {
            k: mx.array(np.stack([sample[k] for sample in batch]))
            for k in batch[0]
        }

config = BERT_Config.from_yaml("./config.yaml")
model = BERT2BERT(config)
model.encoder.load_weights(PRETRAINED_ENCODER_PATH)
optimizer = optim.AdamW(learning_rate=LR)

def seq2seq_loss(logits, labels, padding_mask):
    print(type(logits))
    vocab_size = logits.shape[-1]
    logits = logits.reshape(-1, vocab_size)
    labels = labels.flatten()
    padding_mask = padding_mask.flatten()
    loss = nn.losses.cross_entropy(logits, labels, reduction="none")
    masked_loss = loss * padding_mask
    return mx.mean(masked_loss)

train_losses = []
val_losses = []

def compute_loss(params, model_template, batch):
    model_copy = type(model_template)(model_template.config)
    model_copy.update(params)

    logits = model_copy(batch["encoder_input_ids"], batch["decoder_input_ids"])
    loss = seq2seq_loss(logits, batch["labels"], batch["padding_mask"])
    return loss

loss_and_grad_fn = mx.value_and_grad(compute_loss, argnums=0)

for epoch in range(NUM_EPOCHS):
    epoch_train_loss = 0
    num_train_batches = 0
    batches = create_batches(train_dataset, BATCH_SIZE)

    for batch in tqdm(batches, desc=f"Epoch {epoch+1} - Training"):
        num_train_batches += 1
        loss, grads = loss_and_grad_fn(model.parameters(), model, batch)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)

        epoch_train_loss += float(loss)

        # if step % 50 == 0:
        #     print(f"Epoch {epoch+1}, Step {step}, Train Loss: {float(loss):.4f}")

    avg_train_loss = epoch_train_loss / num_train_batches if num_train_batches > 0 else 0
    train_losses.append(avg_train_loss)
    print(f">>> Epoch {epoch+1} completed, Avg Train Loss: {avg_train_loss:.4f}")

    # Validation
    epoch_val_loss = 0
    num_val_batches = 0
    val_batches = create_batches(val_dataset, BATCH_SIZE)

    for batch in tqdm(val_batches, desc=f"Epoch {epoch+1} - Validation"):
        num_val_batches += 1
        model_copy = type(model)(model.config)
        model_copy.update(model.parameters())
        logits = model_copy(batch["encoder_input_ids"], batch["decoder_input_ids"])
        loss = seq2seq_loss(logits, batch["labels"], batch["padding_mask"])
        epoch_val_loss += float(loss)

    avg_val_loss = epoch_val_loss / num_val_batches if num_val_batches > 0 else 0
    val_losses.append(avg_val_loss)
    print(f">>> Epoch {epoch+1} Validation Loss: {avg_val_loss:.4f}")

plt.figure(figsize=(10, 6))
plt.plot(range(1, NUM_EPOCHS + 1), train_losses, label='Training Loss', marker='o')
plt.plot(range(1, NUM_EPOCHS + 1), val_losses, label='Validation Loss', marker='x')
plt.title('Training and Validation Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

# saving model
model.save_weights("safetensors/bert2bert_decoder.safetensors")