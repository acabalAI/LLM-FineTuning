# Module 3: Tokenization
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset
import torch

# Initialize the tokenizer and model
model_name = 'google/flan-t5-base'
original_model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.float32)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Define a function to tokenize examples
def tokenize_function(example):
    start_prompt = 'Summarize the following conversation.\n'
    end_prompt = '\nSummary:'
    prompt = [start_prompt + dialogue + end_prompt for dialogue in example['dialogue']]
    example['input_ids'] = tokenizer(prompt, padding='max_length', truncation=True, return_tensors='pt').input_ids
    example['labels'] = tokenizer(example['summary'], padding='max_length', truncation=True, return_tensors='pt').input_ids
    return example

# Tokenize the dataset
#tokenized_datasets = dataset.map(tokenize_function, batched=True, batch_size=8)
#tokenized_datasets = tokenized_datasets.remove_columns(['id', 'topic', 'dialogue', 'summary'])
#Reduce dataset (Optional to save resources)
#tokenized_datasets=tokenized_datasets.filter(lambda example,index: index % 100 ==0 , with_indices=True)
