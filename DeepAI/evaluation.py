from pprint import pprint

import datasets
import tempfile
import logging
import random
import config
import os
import yaml
import logging
import difflib
import pandas as pd

import transformers
import datasets
import torch

from tqdm import tqdm

from utilities import *
from transformers import AutoTokenizer, AutoModelForCausalLM

laminiDocsFilename = "lamini_docs.jsonl"
logger = logging.getLogger(__name__)
global_config = None

test_dataset = datasets.load_dataset('json',data_files=laminiDocsFilename, split='train')
trained_model_name = "lamini_docs_10_steps/final"

trained_model = AutoModelForCausalLM.from_pretrained(trained_model_name, local_files_only=True)
trained_tokenizer = AutoTokenizer.from_pretrained(trained_model_name)
trained_tokenizer.pad_token = trained_tokenizer.eos_token

def is_exact_match(a: str, b: str) -> bool:
    return a.strip() == b.strip()

trained_model.eval()

def inference(text, model, tokenizer, max_input_tokens=1000, max_output_tokens=100) -> str:
    # Tokenize
    tokenizer.pad_token = tokenizer.eos_token
    input_ids = tokenizer.encode(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_tokens
    )

    # Generate
    device = model.device
    generated_tokens_with_prompt = model.generate(
        input_ids=input_ids.to(device),
        max_length=max_output_tokens
    )

    # Decode
    generated_text_with_prompt = tokenizer.batch_decode(generated_tokens_with_prompt, skip_special_tokens=True)

    # Strip the prompt
    generated_text_answer = generated_text_with_prompt[0][len(text):]

    return generated_text_answer.strip()


n = 10
metrics = {'exact_matches': []}
predictions = []
for i, item in tqdm(enumerate(test_dataset)):
    # print("i Evaluating: " + str(item))
    question = item['question']
    answer = item['answer']

    try:
        predicted_answer = inference(question, trained_model, trained_tokenizer)
    except:
        continue
    predictions.append([predicted_answer, answer])

    #fixed: exact_match = is_exact_match(generated_answer, answer)
    exact_match = is_exact_match(predicted_answer, answer)
    metrics['exact_matches'].append(exact_match)

    if i > n != -1:
        break
print('Number of exact matches: ', sum(metrics['exact_matches']))

df = pd.DataFrame(predictions, columns=["predicted_answer", "target_answer"])
print(df)