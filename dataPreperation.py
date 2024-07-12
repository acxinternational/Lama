from pprint import pprint

import datasets
from common import laminiDocsFilename
from localTokenizer import tokenize_function


def tokenize_and_split_data():
    fine_tuning_dataset_loaded = datasets.load_dataset("json", data_files=laminiDocsFilename, split="train")

    tokenized_dataset = fine_tuning_dataset_loaded.map(
        tokenize_function,
        batched=True,
        batch_size=1,
        drop_last_batch=True
    )

    tokenized_dataset = tokenized_dataset.add_column("labels", tokenized_dataset["input_ids"])
    tokenized_dataset = tokenized_dataset.train_test_split(test_size=0.2, shuffle=True, seed=98765)

    train_data_set = tokenized_dataset.get("train")
    test_data_set = tokenized_dataset.get("test")

    return train_data_set, test_data_set
