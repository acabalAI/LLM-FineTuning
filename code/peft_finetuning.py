# Module 4: Model Fine-tuning with PEFT
from peft import LoraConfig, get_peft_model, TaskType
import torch
import evaluate
import pandas as pd
import numpy as np
import time

from tqdm import tqdm
tqdm.pandas()

lora_config = LoraConfig(
    r=32,
    lora_alpha=32,
    target_modules=['q', 'v'],
    lora_dropout=0.05,
    bias='none',
    task_type=TaskType.SEQ_2_SEQ_LM
)

peft_model = get_peft_model(original_model, lora_config)

# Continue with fine-tuning
output_dir = f'./peft-dialogue-summary-training-{str(int(time.time()))}'
peft_training_args = TrainingArguments(
    output_dir=output_dir,
    auto_find_batch_size=True,
    learning_rate=1e-3,
    num_train_epochs=10,
    logging_steps=10,
    max_steps=10
)

# Create a trainer for PEFT fine-tuning
peft_trainer = Trainer(
    model=peft_model,
    args=peft_training_args,
    train_dataset=tokenized_datasets['train'],)

# PEFT Training
#peft_trainer.train()
