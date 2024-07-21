import torch
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the dataset
outputDir = "./Nietzsche_Model"
dataset = load_from_disk('../LocalDatasets/beyond_good_and_evil')
'''
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
    f_inputs = tokenizer(examples['prompt'], truncation=True, padding='max_length', max_length=512)
    f_outputs = tokenizer(examples['completion'], truncation=True, padding='max_length', max_length=512)

    f_inputs["labels"] = f_outputs["input_ids"]
    return f_inputs


tokenized_datasets = tokenized_datasets.map(preprocess_function, batched=False, remove_columns=['prompt', 'completion'])

# Initialize the data collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Split the dataset
train_dataset = tokenized_datasets['train']
eval_dataset = tokenized_datasets['test']

# Load the model
model = AutoModelForCausalLM.from_pretrained(checkpoint)

# Define training arguments
training_args = TrainingArguments(
    output_dir=outputDir,
    evaluation_strategy='steps',
    learning_rate=3e-6,
    per_device_train_batch_size=1,  # Set batch size to 1
    per_device_eval_batch_size=1,   # Set batch size to 1
    # num_train_epochs=1,
    max_steps=90,
    weight_decay=0.01,
    save_total_limit=3,
)

# Define the compute metrics function
metric = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    # Flatten the predictions and labels
    predictions = predictions.flatten()
    labels = labels.flatten()

    # Filter out padding tokens (assuming padding token ID is -100)
    mask = labels != tokenizer.pad_token_id
    predictions = predictions[mask]
    labels = labels[mask]

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
outputModelDir = outputDir + '/final'
model.save_pretrained(outputModelDir)
tokenizer.save_pretrained(outputModelDir)
'''
outputModelDir = outputDir + '/final'

# Load the model and tokenizer for inference
model = AutoModelForCausalLM.from_pretrained(outputModelDir)
tokenizer = AutoTokenizer.from_pretrained(outputModelDir)

# Example inference
example_prompt = 'What does Nietzsche mean by the word "human"?'

inputs = tokenizer(example_prompt, return_tensors='pt', truncation=True, padding=True, max_length=512)
with torch.no_grad():
    outputs = model.generate(**inputs, max_length=256, num_beams=2, no_repeat_ngram_size=1, early_stopping=True)
    completion = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(f"Generated completion: {completion}")