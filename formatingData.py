import jsonlines

import pandas as pd
from pprint import pprint

filename = "lamini_docs.jsonl"
instruction_dataset_df = pd.read_json(filename, lines=True)
examples = instruction_dataset_df.to_dict()

prompt_template_qa = """### Question:
{question}

### Answer:
{answer}"""

n = len(examples["question"])
fine_tuning_dataset_text_only = []
fine_tuning_dataset_question_answer = []

for i in range(n):
    q = examples["question"][i]
    a = examples["answer"][i]
    text_with_prompt_template_qa = prompt_template_qa.format(question=q, answer=a)
    fine_tuning_dataset_question_answer.append({"text": text_with_prompt_template_qa})

with jsonlines.open(f'lamini_docs_processed.jsonl', 'w') as writer:
    writer.write_all(fine_tuning_dataset_question_answer)

pprint(fine_tuning_dataset_question_answer)


