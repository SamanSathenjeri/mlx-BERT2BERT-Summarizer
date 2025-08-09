import os
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import mlx.utils as utils
from tqdm import tqdm
from itertools import chain
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import json

from datasets import load_dataset, concatenate_datasets
from transformers import BertTokenizerFast
from BERTEncoder import BERT, BERT_Config, load_config

def compute_loss(params, model_instance_for_forward, inputs, targets):
    model_copy = type(model_instance_for_forward)(model_instance_for_forward.config)
    model_copy.update(params)

    logits = model_copy(inputs)
    losses = nn.losses.cross_entropy(logits, targets, reduction='none')

    mask = targets != -100
    masked_losses = losses * mask.astype(losses.dtype)

    num_valid = mask.sum()
    loss = masked_losses.sum() / num_valid
    return loss

def evaluate(model, val_data, batch_size):
    total_loss = 0
    num_batches = val_data['input_ids'].shape[0] // batch_size

    for i in tqdm(range(num_batches), desc=f"Validation"):
        start = i * batch_size
        end = start + batch_size
        batch_input = val_data['input_ids'][start:end]
        batch_labels = val_data['labels'][start:end]

        loss = compute_loss(model.parameters(), model, batch_input, batch_labels)
        total_loss += loss.item()

    return total_loss / num_batches

def train(model, train_data, val_data, batch_size, num_epochs, learning_rate):
    train_loss_list = []
    val_loss_list = []
    loss_and_grad_fn = mx.value_and_grad(compute_loss, argnums=0) 
    optimizer = optim.AdamW(learning_rate=learning_rate)

    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = train_data['input_ids'].shape[0] // batch_size

        # Compile the training step for efficiency
        def compiled_train_step(current_model_parameters, model_instance, inputs, targets):
            loss, grads = loss_and_grad_fn(current_model_parameters, model_instance, inputs, targets)
            optimizer.update(model_instance, grads) 
            
            return loss

        for i in tqdm(range(num_batches), desc=f"Epoch {epoch+1}"):
            start = i * batch_size
            end = start + batch_size
            batch_input = train_data['input_ids'][start:end]
            batch_labels = train_data['labels'][start:end]

            loss = compiled_train_step(model.parameters(), model, batch_input, batch_labels)
            mx.eval(model.parameters(), optimizer.state) 

            total_loss += loss.item()

        avg_loss = total_loss / num_batches
        val_loss = evaluate(model, val_data, batch_size)
        print(f"Epoch {epoch+1} avg train loss: {avg_loss:.4f} | val loss: {val_loss:.4f}")

        train_loss_list.append(avg_loss)
        val_loss_list.append(val_loss)

        if epoch % 2 == 0:
            mx.eval(model.parameters())
            flat_weights = dict(utils.tree_flatten(model.parameters()))
            print("Saving these keys:", flat_weights.keys())
            output_file_path = "safetensors/BERT_weights.safetensors"

            metadata = {
                "model_type": "BERT",
                "framework": "MLX",
                "description": "MLX BERT model weights",
                "date_saved": "2025-07-28"
            }

            for k, v in flat_weights.items():
                print(f"{k}: type={type(v)} shape={getattr(v, 'shape', None)}")

            try:
                mx.save_safetensors(output_file_path, flat_weights, metadata=metadata)
                print(f"MLX BERT model weights successfully saved to {output_file_path}")
            except Exception as e:
                print(f"Error saving MLX model weights: {e}")

    loss_log = {
        "train_loss": train_loss_list, 
        "val_loss": val_loss_list,      
    }

    with open("loss_log.json", "w") as f:
        json.dump(loss_log, f)

    plt.figure(figsize=(10, 6))
    plt.plot(train_loss_list, label="Training Loss")
    plt.plot(val_loss_list, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("BERT Pretraining Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("loss_curve.png")
    plt.close()

if __name__ == "__main__":
    config = BERT_Config.from_yaml()
    model = BERT(config)

    WEIGHTS_PATH = "safetensors/BERT_weights.safetensors"
    if os.path.exists(WEIGHTS_PATH):
        model.load_weights(WEIGHTS_PATH)

    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

    TRAIN_PREPROCESSED_PATH = "safetensors/combined_mlm_preprocessed.safetensors"
    VALIDATION_PREPROCESSED_PATH = "safetensors/val_combined_mlm_preprocessed.safetensors"
    
    if not os.path.exists(TRAIN_PREPROCESSED_PATH) or not os.path.exists(VALIDATION_PREPROCESSED_PATH):
        print("Loading datasets...")
        wiki2 = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        cnn_dm = load_dataset("cnn_dailymail", "3.0.0", split="train[:10%]")

        # Take only the article field from CNN/DM
        cnn_articles = cnn_dm.map(lambda x: {"text": x["article"]}, remove_columns=cnn_dm.column_names)

        # Combine datasets
        combined_dataset = concatenate_datasets([wiki2, cnn_articles])
        print(f"Combined dataset size: {len(combined_dataset)}")
        # texts = combined_dataset["text"]
        texts = list(combined_dataset["text"])
        train_texts, val_texts = train_test_split(texts, test_size=0.05, random_state=42)

        block_size = 128

        def chunkify(lst, size):
            return [lst[i:i+size] for i in range(0, len(lst) - size, size)]

        # Tokenize and flatten into a long sequence
        def tokenize_and_chunk(text_list, tokenizer, block_size):
            input_ids = []
            for i in tqdm(range(0, len(text_list), 1000), desc="Tokenizing"):
                batch = text_list[i:i + 1000]
                tokenized = tokenizer(batch, return_special_tokens_mask=True, truncation=True, padding=False)
                for ids in tokenized["input_ids"]:
                    input_ids.extend(ids)
            input_ids = [i for i in input_ids if i != tokenizer.pad_token_id]
            return chunkify(input_ids, block_size)

        print("Chunking...")
        train_chunks = tokenize_and_chunk(train_texts, tokenizer, block_size)
        val_chunks = tokenize_and_chunk(val_texts, tokenizer, block_size)

        train_chunks = mx.array(train_chunks, dtype=mx.int32)
        val_chunks = mx.array(val_chunks, dtype=mx.int32)

        # MLM-style masking function
        def mask_tokens(inputs, tokenizer, mlm_probability=0.15):
            labels = mx.full(inputs.shape, vals=-100, dtype=mx.int32)
            probability_matrix = mx.random.uniform(shape=inputs.shape)
            mask_arr = probability_matrix < mlm_probability
            labels = mx.where(mask_arr, inputs, labels)

            rand = mx.random.uniform(shape=inputs.shape)
            mask_token_id = tokenizer.mask_token_id

            mask_replace = mask_arr & (rand < 0.8)
            random_replace = mask_arr & (rand >= 0.8) & (rand < 0.9)

            inputs_modified = mx.where(mask_replace, mask_token_id, inputs)

            num_replacements = random_replace.sum().item()
            if num_replacements > 0:
                random_vals = mx.random.randint(low=0, high=tokenizer.vocab_size, shape=inputs.shape, dtype=mx.int32)
                inputs_final = mx.where(random_replace, random_vals, inputs_modified)
            else:
                inputs_final = inputs_modified

            return inputs_final, labels

        print("Applying MLM masking...")
        train_input_masked, train_labels = mask_tokens(train_chunks, tokenizer)
        val_input_masked, val_labels = mask_tokens(val_chunks, tokenizer)

        train_data = {
            "input_ids": train_input_masked,
            "labels": train_labels
        }
        val_data = {
            "input_ids": val_input_masked,
            "labels": val_labels
        }

        print("Saving preprocessed dataset...")
        mx.save_safetensors(TRAIN_PREPROCESSED_PATH, train_data)
        mx.save_safetensors(VALIDATION_PREPROCESSED_PATH, val_data)

    else:
        print("Loading preprocessed dataset")
        train_data = mx.load(TRAIN_PREPROCESSED_PATH)
        val_data = mx.load(VALIDATION_PREPROCESSED_PATH)

    print("starting to train")

    dummy_input = mx.zeros((1, config.block_size), dtype=mx.int32)
    _ = model(dummy_input)

    def load_train_configs(path="config.yaml"):
        config = load_config(path)
        return config['training']['batch_size'], config['training']['num_epochs'], config['training']['learning_rate']

    batch_size, num_epochs, learning_rate = load_train_configs()
    steps_per_epoch = train_data['input_ids'].shape[0] // batch_size
    lr_schedule = optim.cosine_decay(learning_rate, num_epochs*steps_per_epoch)

    train(model, train_data, val_data, batch_size=batch_size, num_epochs=num_epochs, learning_rate=lr_schedule)

    # mx.eval(model.parameters())
    # flat_weights = dict(utils.tree_flatten(model.parameters()))
    # print("Saving these keys:", flat_weights.keys())
    # output_file_path = "safetensors/BERT_weights.safetensors"

    # metadata = {
    #     "model_type": "BERT",
    #     "framework": "MLX",
    #     "description": "MLX BERT model weights",
    #     "date_saved": "2025-07-28"
    # }

    # for k, v in flat_weights.items():
    #     print(f"{k}: type={type(v)} shape={getattr(v, 'shape', None)}")

    # try:
    #     mx.save_safetensors(output_file_path, flat_weights, metadata=metadata)
    #     print(f"MLX BERT model weights successfully saved to {output_file_path}")
    # except Exception as e:
    #     print(f"Error saving MLX model weights: {e}")