# Fine-Tuning LLM with PEFT for Text Summarization

This project involves the fine-tuning of a Language Model using Parameter-Efficient Fine-Tuning (PEFT) for the purpose of text summarization. The goal is to create an efficient and specialized summarization model that can generate concise and coherent summaries of input text while efficiently using the resources at our disposal.
For that purpouse a series of PEFT techniques are implemented, the final result is evaluated using ROUGE metric.
In this project we can see how even using a fraction of the resources needed for a traditional fine-tuning system we can achive remarkable improvements compared to the baseline model.

## Table of Contents

- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Methodology](#methodology)
- [Code Structure](#code-structure)
- [Usage](#usage)
- [Results](#results)
- [Reference](#reference)


## Getting Started

### Prerequisites

Before using this project, ensure you have the following prerequisites:
```bash
Python 3.x
PyTorch
Transformers library
Hugging Face Datasets library
Accelerate library
```

You can install the dataset using the following commands:

```bash
! pip install transformers
! pip install datasets
! pip install torch
! pip install torchdata
! pip install accelerate -U
! pip install evaluate
! pip install rouge_score
! pip install loralib
! pip install peft
! pip install trl
```

### Installation

```bash
git clone https://github.com/yourusername/peft-summarization.git
cd peft-summarization
```

## Methodology

### Data Preprocessing

  - Tokenization of input text and target summaries.
  - Dataset formatting and splitting for fine-tuning.

### Fine-Tuning with PEFT

  - Utilization of PEFT to enhance the base Language Model for summarization.
  - Training of the PEFT model using specific configurations.

## Evaluation

- Evaluation of the fine-tuned model's performance using evaluation metrics such as ROUGE.

## Code Structure

The project code is organized into different sections:

    preprocessing.py/: Code for data preprocessing and dataset loading.
    build_database.py/: Code to create the database, splitting between training and testing set
    tokenization.py/:Code to tokenize and prepare the data for ingestion.
    PEFT-finetuning.py/:Code to fine-tune the model using PEFT techniques
    evaluate.py/: Code for evaluating model performance.
    main.py/: function contains the main logic of the program, and it's executed when you run main.py directly.

## Usage

To fine-tune the PEFT model and generate text summaries, follow the instructions and code examples in the provided scripts and notebooks.
The user can explore the size of the dataset to fine-tune as well as twisting the hyperparameters to explore different architectural alternatives.

## Results

In this project, we explored the use of Parameter-Efficient Fine-Tuning (PEFT) as a resource-effective fine-tuning technique. Our exploration encompassed data preprocessing, base model training, and PEFT fine-tuning, with a focus on result comparison. The performance metrics underscore the remarkable potential of PEFT, as the fine-tuned model outperformed the original across all ROUGE metrics. Notably, the PEFT fine-tuned model achieved substantial improvements, with ROUGE-1 increasing to 0.37253, ROUGE-2 to 0.121388, ROUGE-L to 0.27054, and ROUGE-Lsum to 0.2758. These results affirm the remarkable ability of PEFT to elevate text summarization quality, even with limited compute and using just a fraction of the entire database for training.
A more in-depth report is available in the folder results.

## Reference

```bibtex
@Misc{peft,
  title = {PEFT: State-of-the-art Parameter-Efficient Fine-Tuning methods},
  author = {Sourab Mangrulkar and Sylvain Gugger and Lysandre Debut and Younes Belkada and Sayak Paul and Benjamin Bossan},
  howpublished = {\url{https://github.com/huggingface/peft}},
  year = {2022}
}
