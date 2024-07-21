from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
import evaluate
import numpy as np
import torch

# Load the dataset
dataset = load_from_disk('../LocalDatasets/beyond_good_and_evil')

# Initialize the tokenizer
checkpoint = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
tokenizer.pad_token = tokenizer.eos_token

# Tokenize the dataset
def tokenize_function(example):
    return tokenizer(example['prompt'], example['completion'], truncation=True, padding='max_length', max_length=512)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Prepare the dataset for training
def preprocess_function(examples):
    inputs = tokenizer(examples['prompt'], truncation=True, padding='max_length', max_length=512)
    outputs = tokenizer(examples['completion'], truncation=True, padding='max_length', max_length=512)

    inputs["labels"] = outputs["input_ids"]
    return inputs

tokenized_datasets = tokenized_datasets.map(preprocess_function, batched=True, remove_columns=['prompt', 'completion'])

# Initialize the data collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Split the dataset
train_dataset = tokenized_datasets['train']
eval_dataset = tokenized_datasets['test']

# Load the model
model = AutoModelForCausalLM.from_pretrained(checkpoint)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy='steps',
    learning_rate=2e-5,
    per_device_train_batch_size=2,  # Set batch size to 1
    per_device_eval_batch_size=2,   # Set batch size to 1
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=2,
)

# Define the compute metrics function
metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()

# Save the model and tokenizer
model.save_pretrained('./saved_model')
tokenizer.save_pretrained('./saved_model')

# Load the model and tokenizer for inference
model = AutoModelForCausalLM.from_pretrained('./saved_model')
tokenizer = AutoTokenizer.from_pretrained('./saved_model')

# Example inference
prompt = "What is the main idea of Nietzsche's philosophy?"

inputs = tokenizer(prompt, return_tensors='pt', truncation=True, padding=True, max_length=512)
with torch.no_grad():
    outputs = model.generate(**inputs, max_length=100, num_beams=5, early_stopping=True)
    completion = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(f"Generated completion: {completion}")
