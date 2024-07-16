from common import tokenizer


def tokenize_function(examples):
    if "question" in examples and "answer" in examples:
        text = examples["question"][0] + examples["answer"][0]
    elif "input" in examples and "output" in examples:
        text = examples["input"][0] + examples["output"][0]
    else:
        text = examples["text"][0]

    tokenized_inputs = tokenizer(
        text,
        return_tensors="np",
        padding=True,
    )

    max_length = min(
        tokenized_inputs["input_ids"].shape[1],
        2048
    )

    if tokenized_inputs["input_ids"].shape[1] > max_length:
        print(f"Truncating input from {tokenized_inputs['input_ids'].shape[1]} to {max_length}")

    tokenizer.truncation_side = "left"

    tokenized_inputs = tokenizer(
        text,
        return_tensors="np",
        truncation=True,
        max_length=max_length
    )

    return tokenized_inputs
