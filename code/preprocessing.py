# Module 1: Data Preparation and Preprocessing
from datasets import load_dataset
huggingface_dataset_name = 'knkarthick/dialogsum'
input_min_text_length = 200
input_max_text_length = 1000

def load_original_dataset(dataset_name):
    return load_dataset(dataset_name)

def build_dataset(original_dataset, model_name, input_min_text_length, input_max_text_length):
    dataset = original_dataset['train']
    dataset = dataset.filter(lambda x: len(x['dialogue']) > input_min_text_length and len(x['dialogue']) <= input_max_text_length)
    return dataset

# Define the dataset name
#dataset= load_original_dataset(huggingface_dataset_name)
