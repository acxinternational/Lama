import jsonlines

import pandas as pd
from pprint import pprint
from common import processedDataFileName
from common import laminiDocsFilename


instruction_dataset_df = pd.read_json(laminiDocsFilename, lines=True)
examples = instruction_dataset_df.to_dict()

prompt_template_qa = '{{ "input": "{question}", "output": "{answer}" }}'

n = len(examples["question"])
fine_tuning_dataset_text_only = []
fine_tuning_dataset_question_answer = []

for i in range(n):
    q = examples["question"][i].replace("'", "").replace('"', "").replace(',', "")
    a = examples["answer"][i].replace("'", "").replace('"', "").replace(',', "")
    text_with_prompt_template_qa = prompt_template_qa.format(question=q, answer=a)
    fine_tuning_dataset_question_answer.append(text_with_prompt_template_qa)

with jsonlines.open(processedDataFileName, 'w') as writer:
    writer.write_all(fine_tuning_dataset_question_answer)

pprint(fine_tuning_dataset_question_answer)


