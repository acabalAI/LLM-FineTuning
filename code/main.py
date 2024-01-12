if __name__ == "__main__":

  device = torch.device("cuda:0")

  dataset_original = load_original_dataset(huggingface_dataset_name)

  dataset=build_dataset(model_name=model_name,dataset_name=huggingface_dataset_name,input_min_text_length=200,
                        input_max_text_length=1000)
  #Tokenize the dataset
  tokenized_datasets = dataset.map(tokenize_function, batched=True, batch_size=8)
  tokenized_datasets = tokenized_datasets.remove_columns(['id', 'topic', 'dialogue', 'summary'])
  #Reduce dataset (Optional to save resources)
  tokenized_datasets=tokenized_datasets.filter(lambda example,index: index % 100 ==0 , with_indices=True)

  peft_trainer = Trainer(
    model=peft_model,
    args=peft_training_args,
    train_dataset=tokenized_datasets['train'],)
  
  peft_trainer.train()

  dataset = load_dataset(huggingface_dataset_name)
  dialogues = dataset['test'][0:100]['dialogue']
  human_baseline_summaries = dataset['test'][0:100]['summary']

  original_model_results, peft_model_results = evaluate_model(original_model, peft_model, dialogues, tokenizer, rouge)

  # Print or store the evaluation results
  print('Original Model:')
  print(original_model_results)

  print('PEFT Model:')
  print(peft_model_results)
