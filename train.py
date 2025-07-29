import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import mlx.utils as utils
from tqdm import tqdm

from datasets import load_dataset, concatenate_datasets
from transformers import BertTokenizerFast
from BERTEncoder import BERT, BERT_Config

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

def train(model, train_data, batch_size, num_epochs, learning_rate):
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
        print(f"Epoch {epoch+1} avg loss: {avg_loss:.4f}")

if __name__ == "__main__":
    config = BERT_Config.from_yaml()
    model = BERT(config)
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

    def chunkify(lst, size):
        return [lst[i:i+size] for i in range(0, len(lst)-size, size)]

    chunks = chunkify(input_ids, block_size)
    chunks = mx.array(chunks, dtype=mx.int32)
    
    def mask_tokens(inputs, tokenizer, mlm_probability=0.15):
        labels = mx.full(inputs.shape, vals=-100, dtype=mx.int32)

        probability_matrix = mx.random.uniform(shape=inputs.shape)
        mask_arr = probability_matrix < mlm_probability
        labels = mx.where(mask_arr, inputs, labels)

        rand = mx.random.uniform(shape=inputs.shape)
        mask_token_id = tokenizer.mask_token_id

        mask_replace = mask_arr & (rand < 0.8)
        random_replace = mask_arr & (rand >= 0.8) & (rand < 0.9)
        unchanged = mask_arr & (rand >= 0.9)

        inputs_modified = mx.where(mask_replace, mask_token_id, inputs)
        num_replacements = random_replace.sum().item()

        if num_replacements > 0:
            full_random_values = mx.random.randint(low=0, high=tokenizer.vocab_size, shape=inputs.shape, dtype=mx.int32)
            inputs_final = mx.where(random_replace, full_random_values, inputs_modified)
        else:
            inputs_final = inputs_modified

        return inputs_final, labels

    input_ids_masked, labels = mask_tokens(mx.array(chunks), tokenizer) 

    train_data = {
        "input_ids": input_ids_masked,
        "labels": labels
    }

    print("starting to train")

    dummy_input = mx.zeros((1, config.block_size), dtype=mx.int32)
    _ = model(dummy_input)

    train(model, train_data, batch_size=1, num_epochs=10, learning_rate=1e-4)

    mx.eval(model.parameters())
    flat_weights = dict(utils.tree_flatten(model.parameters()))
    print("Saving these keys:", flat_weights.keys())
    output_file_path = "BERT_weights.safetensors"

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

# epoch 1 = 7.1007
# epoch 2 = 6.7244
# epoch 3 = 6.5495
