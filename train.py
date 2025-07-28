import math
import yaml
import numpy as np
from dataclasses import dataclass
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import mlx.utils as utils

from transformers import BertTokenizer
from model import BERT, BERT_Config

if __name__ == "__main__":
    config = BERT_Config()
    model = BERT(config)

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    text = "this is a test"
    inputs = tokenizer(text, return_tensors="pt", padding="max_length", max_length=64, truncation=True)

    input_ids = inputs["input_ids"]  # shape: [1, 64]
    attention_mask = inputs["attention_mask"]

    logits = model(input_ids)
    print(logits)