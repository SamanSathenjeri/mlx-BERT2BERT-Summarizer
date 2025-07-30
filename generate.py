import mlx.core as mx
import numpy as np
from transformers import AutoTokenizer
from datasets import load_dataset
from BERTEncoder import BERT_Config
from BERTDecoder import BERT2BERT, make_padding_mask
import evaluate
import os

# Constants
MAX_GEN_LEN = 64

rouge = evaluate.load("rouge")
config = BERT_Config.from_yaml("config.yaml")
model = BERT2BERT(config)
model.load_weights("bert2bert_decoder.safetensors")
# model.eval()

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
tokenizer.pad_token = tokenizer.eos_token or tokenizer.sep_token or "[PAD]"

PAD_TOKEN_ID = tokenizer.pad_token_id

def decode_tokens(token_ids):
    return tokenizer.decode(token_ids, skip_special_tokens=True)

def beam_search(encoder_output, encoder_mask, model, tokenizer, beam_width=3, max_length=64):
    sequences = [([tokenizer.cls_token_id], 0)]

    for _ in range(max_length):
        all_candidates = []

        for seq, score in sequences:
            decoder_input = mx.array([seq])
            decoder_mask = make_padding_mask(decoder_input, pad_token_id=PAD_TOKEN_ID)

            logits = model.decoder(
                decoder_input,
                encoder_output,
                decoder_pad_mask=decoder_mask,
                encoder_pad_mask=encoder_mask,
            )

            next_token_logits = logits[:, -1, :]
            next_token_logprobs = mx.log_softmax(next_token_logits, axis=-1).squeeze(0).tolist()

            for token_id, logprob in enumerate(next_token_logprobs):
                candidate_seq = seq + [token_id]
                candidate_score = score + logprob
                all_candidates.append((candidate_seq, candidate_score))

        ordered = sorted(all_candidates, key=lambda tup: tup[1], reverse=True)
        sequences = ordered[:beam_width]

        if all(seq[-1] in {tokenizer.eos_token_id, tokenizer.sep_token_id} for seq, _ in sequences):
            break

    best_seq = sequences[0][0][1:]
    return decode_tokens(best_seq)

def generate_summary(text: str, max_length=MAX_GEN_LEN, use_beam_search=False):
    encoder_inputs = tokenizer(
        text,
        return_tensors="np",
        truncation=True,
        max_length=config.block_size,
        padding="max_length",
    )
    encoder_input_ids = mx.array(encoder_inputs["input_ids"])
    B, T = encoder_input_ids.shape

    encoder_mask = make_padding_mask(encoder_input_ids, pad_token_id=tokenizer.pad_token_id)
    encoder_output = model.encoder(encoder_input_ids, mask=encoder_mask, return_embeddings=True)

    if model.freeze_encoder_weights:
        encoder_output = mx.stop_gradient(encoder_output)

    if use_beam_search:
        return beam_search(encoder_output, encoder_mask, model, tokenizer, max_length=max_length)

    # Original greedy decoding
    decoder_input = mx.array([[tokenizer.cls_token_id]])

    for _ in range(max_length):
        decoder_mask = make_padding_mask(decoder_input, pad_token_id=PAD_TOKEN_ID)
        logits = model.decoder(
            decoder_input,
            encoder_output,
            decoder_pad_mask=decoder_mask,
            encoder_pad_mask=encoder_mask,
        )
        next_token_logits = logits[:, -1, :]
        next_token = mx.argmax(next_token_logits, axis=-1)
        decoder_input = mx.concatenate([decoder_input, next_token[:, None]], axis=1)

        if next_token[0].item() in {tokenizer.sep_token_id, tokenizer.eos_token_id}:
            break

    generated_ids = decoder_input[0].tolist()[1:]  # Remove [CLS]
    return decode_tokens(generated_ids)

def evaluate_model_on_dataset(dataset, max_examples=100):
    predictions = []
    references = []

    for i, sample in enumerate(dataset[:max_examples]):
        input_text = sample["input_text"]
        reference = sample["target_text"]
        predicted = generate_summary(input_text)
        predictions.append(predicted)
        references.append(reference)

    scores = rouge.compute(predictions=predictions, references=references)
    return scores

def interactive_loop():
    print("Enter text to summarize, type 'eval' to evaluate, 'file <path>' to upload a file, or 'exit'.")

    while True:
        user_input = input("\n> ").strip()

        if user_input.lower() == "exit":
            print("👋 Exiting.")
            break
        elif user_input.lower() == "eval":
            print("📥 Loading CNN/DailyMail dataset...")
            dataset = load_dataset("cnn_dailymail", "3.0.0", split="validation[:1000]")
            eval_data = [{"input_text": x["article"], "target_text": x["highlights"]} for x in dataset]

            print("📊 Running evaluation...")
            scores = evaluate_model_on_dataset(eval_data)
            print("✅ ROUGE Scores:", scores)
        elif user_input.lower().startswith("file "):
            file_path = user_input.split(" ", 1)[1]
            if not os.path.isfile(file_path):
                print("❌ File not found.")
                continue
            with open(file_path, "r", encoding="utf-8") as f:
                file_text = f.read()
            print(f"\n📄 Summarizing {file_path}...")
            summary = generate_summary(file_text, use_beam_search=True)
            print("\n📝 Summary:\n", summary)
            print("-" * 40)
        else:
            summary = generate_summary(user_input, use_beam_search=True)
            print("\n📝 Summary:\n", summary)
            print("-" * 40)

if __name__ == "__main__":
    interactive_loop()
