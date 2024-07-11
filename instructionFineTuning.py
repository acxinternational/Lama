import lamini

import itertools
import jsonlines

from datasets import load_dataset
from pprint import pprint

from llama import BasicModelRunner
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

dataset_path_hf = "lamini/alpaca"
dataset_hf = load_dataset(dataset_path_hf)
lamini.api_key = "ceefc6b819fc7fb17e02ea7332283f301e5d6b02a38bd67e1ac360cbf8615dc1"

non_instructed_model = BasicModelRunner("meta-llama/Llama-2-7b-hf")
non_instructed_output = non_instructed_model("Tell me how to train my dog to sit")
pprint("Uninstructed output\n" + non_instructed_output)

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")
model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-70m")

def inference(text, model, tokenizer, max_input_tokens=1000, max_output_tokens=100):
    # Tokenize
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

    return generated_text_answer

finetuning_dataset_path = "lamini/lamini_docs"
finetuning_dataset = load_dataset(finetuning_dataset_path)
pprint(finetuning_dataset["test"][0])