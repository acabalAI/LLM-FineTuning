# Module 5: Evaluate with ROUGE
import pandas as pd
from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM
from transformers import AutoTokenizer
from transformers import GenerationConfig
rouge=evaluate.load('rouge')

# Load the test dataset
dataset = load_dataset(huggingface_dataset_name)
dialogues = dataset['test'][0:15]['dialogue']
human_baseline_summaries = dataset['test'][0:15]['summary']


def evaluate_model(original_model, peft_model, dataset, tokenizer, rouge):
    original_model_summaries = []
    peft_model_summaries = []

    for _, dialogue in enumerate(dialogues):
      prompt = f"""
      Summarize the following conversation.
      {dialogue}
      Summary: """

      input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device)

      original_model_outputs = original_model.generate(input_ids=input_ids, generation_config=GenerationConfig(max_new_tokens=200))
      original_model_text_output = tokenizer.decode(original_model_outputs[0], skip_special_tokens=True)
      original_model_summaries.append(original_model_text_output)

      peft_model.to(device)
      peft_model_outputs = peft_model.generate(input_ids=input_ids, generation_config=GenerationConfig(max_new_tokens=200))
      peft_model_text_output = tokenizer.decode(peft_model_outputs[0], skip_special_tokens=True)
      peft_model_summaries.append(peft_model_text_output)

  # Combine and display the results
    zipped_summaries = list(zip(human_baseline_summaries, original_model_summaries, peft_model_summaries))
    df = pd.DataFrame(zipped_summaries, columns=['Human baseline', 'original', 'peft'])

    original_model_results=rouge.compute(predictions=original_model_summaries,
                                        references=human_baseline_summaries[0:len(original_model_summaries)],
                                        use_aggregator=True,
                                        use_stemmer=True,
                                        )
    peft_model_results=rouge.compute(predictions=peft_model_summaries,
                                        references=human_baseline_summaries[0:len(peft_model_summaries)],
                                        use_aggregator=True,
                                        use_stemmer=True,
                                      )

    return original_model_results, peft_model_results


