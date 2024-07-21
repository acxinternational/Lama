from datasets import load_from_disk, load_metric
from transformers import AutoTokenizer, DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
import numpy as np
import evaluate

# Load the dataset from disk
samples = 3
raw_datasets = load_from_disk('../LocalDatasets/glue_mrpc')

train_datasets = raw_datasets['train']

train_data0_sentence1 = train_datasets['sentence1'][0]
train_data0_sentence2 = train_datasets['sentence2'][0]
train_data0_Label = train_datasets['label'][0]
train_data0_idx = train_datasets['idx'][0]

checkpoint = "bert-base-cased"
local_tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# Tokenize function
def tokenize_function(example):
    return local_tokenizer(example['sentence1'], example['sentence2'], padding="max_length", truncation=True, max_length=128)

# Apply the tokenize function to the dataset
tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

# Prepare the dataset for training
tokenized_datasets = tokenized_datasets.remove_columns(['idx', 'sentence1', 'sentence2'])
tokenized_datasets = tokenized_datasets.rename_column('label', 'labels')
tokenized_datasets = tokenized_datasets.with_format('torch')

# Use the correct data collator
local_data_collector = DataCollatorWithPadding(tokenizer=local_tokenizer)

small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))

local_model = AutoModelForSequenceClassification.from_pretrained("google-bert/bert-base-cased", num_labels=5)
training_args = TrainingArguments(
    output_dir="test_trainer",
    evaluation_strategy="epoch",
    max_steps=samples,
)
evaluation_metric = evaluate.load("accuracy")

# Compute metrics function
def compute_metrics(eval_prediction):
    logits, labels = eval_prediction
    predictions = np.argmax(logits, axis=-1)
    return evaluation_metric.compute(predictions=predictions, references=labels)

# Initialize the Trainer
trainer = Trainer(
    model=local_model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    data_collator=local_data_collector,
    tokenizer=local_tokenizer,
    compute_metrics=compute_metrics,
)

# Train the model
result = trainer.train()

# Predict and print results
predictions = trainer.predict(tokenized_datasets["validation"])
print(predictions.predictions.shape, predictions.label_ids.shape)

# Save the trained model and tokenizer
trained_model_name = f"google_bert_{samples}_steps"
save_dir = f'{trained_model_name}/final'

trainer.save_model(save_dir)
local_tokenizer.save_pretrained(save_dir)
local_tokenizer.save_vocabulary(save_dir)
