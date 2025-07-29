import mlx.core as mx
from transformers import AutoTokenizer
from BERTEncoder import BERT_Config
from BERTDecoder import BERT2BERT

MAX_GEN_LEN = 64
DEVICE_PAD_TOKEN = 0

config = BERT_Config.from_yaml("config.yaml")
model = BERT2BERT(config)
model.load_weights("checkpoints/bert2bert_decoder.safetensors")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
tokenizer.pad_token = tokenizer.eos_token or tokenizer.sep_token or "[PAD]"

def generate_summary(text: str):
    encoder_inputs = tokenizer(text, return_tensors="np", truncation=True, max_length=config.block_size, padding="max_length")
    encoder_input_ids = mx.array(encoder_inputs["input_ids"])
    encoder_output = model.encoder(encoder_input_ids)

    B, T = encoder_input_ids.shape
    decoder_input = mx.array([[tokenizer.cls_token_id]])

    for _ in range(MAX_GEN_LEN):
        decoder_padded = mx.pad(decoder_input, ((0, 0), (0, config.block_size - decoder_input.shape[1])), constant_values=DEVICE_PAD_TOKEN)
        logits = model.decoder(decoder_padded, encoder_output)
        next_token_logits = logits[:, decoder_input.shape[1] - 1, :]
        next_token = mx.argmax(next_token_logits, axis=-1)
        decoder_input = mx.concatenate([decoder_input, next_token[:, None]], axis=-1)
        if next_token[0].item() in {tokenizer.sep_token_id, tokenizer.eos_token_id}:
            break

    decoded_ids = decoder_input[0].tolist()[1:]
    summary = tokenizer.decode(decoded_ids, skip_special_tokens=True)
    return summary

if __name__ == "__main__":
    while True:
        user_input = input("> ")
        
        if user_input.lower() == 'exit':
            print("Exiting the program.")
            break

        summary = generate_summary(user_input)
        padding_width = 8
        print(f"{summary:>{padding_width}}")
