import logging

import torch
from transformers import AutoModelForCausalLM
from transformers import Trainer
from transformers import TrainingArguments

from common import *
from dataPreperation import tokenize_and_split_data

logger = logging.getLogger(__name__)
global_config = None

training_config = {
    "model": {
        "pretrained_name": model_pythia_70m_name,
        "max_length" : 2048
    },
    "datasets": {
        "use_hf": False,
        "path": processedDataFileName
    },
    "verbose": True
}

train_dataset, test_dataset = tokenize_and_split_data()
base_model = AutoModelForCausalLM.from_pretrained(model_pythia_70m_name)

device_count = torch.cuda.device_count()
if device_count > 0:
    print("Selected GPU device")
    device = torch.device("cuda")
else:
    print("Selected CPU device")
    device = torch.device("cpu")

base_model.to(device)

dataset_Inx = 33
test_text = test_dataset[dataset_Inx]['question']

max_steps = 3
trained_model_name = f"lamini_docs_{max_steps}_steps"
output_dir = trained_model_name

training_args = TrainingArguments(

    # Learning rate
    learning_rate=1.0e-4,

    # Number of training epochs
    num_train_epochs=1,

    # Max steps to train for (each step is a batch of data)
    # Overrides num_train_epochs, if not -1
    max_steps=max_steps,

    # Batch size for training
    per_device_train_batch_size=1,

    # Directory to save model checkpoints
    output_dir=output_dir,

    # Other arguments
    overwrite_output_dir=False, # Overwrite the content of the output directory
    disable_tqdm=False, # Disable progress bars
    eval_steps=120, # Number of update steps between two evaluations
    save_steps=120, # After # steps model is saved
    warmup_steps=0, # Number of warmup steps for learning rate scheduler
    per_device_eval_batch_size=1, # Batch size for evaluation
    eval_strategy="steps",
    logging_strategy="steps",
    logging_steps=1,
    optim="adafactor",
    gradient_accumulation_steps = 8,
    gradient_checkpointing=False,

    # Parameters for early stopping
    load_best_model_at_end=True,
    save_total_limit=2,
    metric_for_best_model="eval_loss",
    greater_is_better=False
)

model_flops = (
        base_model.floating_point_ops(
            {
                "input_ids": torch.zeros(
                    (1, training_config["model"]["max_length"])
                )
            }
        )
        * training_args.gradient_accumulation_steps
)

trainer = Trainer(
    model=base_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    # data_collator=data_collator   # Ensure this collator returns attention_mask
)

training_output = trainer.train()

save_dir = f'{output_dir}/final'

trainer.save_model(save_dir)

fine_tuned_slightly_model = AutoModelForCausalLM.from_pretrained(save_dir, local_files_only=True)
fine_tuned_slightly_model.to(device)

test_question = test_dataset[dataset_Inx]['question']
print("Question input (test):", test_question)

print("Finetuned slightly model's answer: ", inference(test_question, fine_tuned_slightly_model))

test_answer = test_dataset[dataset_Inx]['answer']
print("Target answer output (test):", test_answer)

