# LLM Fine-tuning for Question-Answering (Q&A) Tasks

## Overview
This project fine-tunes Llama-3.2-3B-Instruct model for domain-specific question-answering tasks to improve accuracy, relevance, and response quality.

## Problem Statement
Large Language Models often struggle with domain-specific question-answering tasks due to lack of contextual grounding. This project aims to enhance model accuracy through fine-tuning on a structured dataset containing prompts, context, and corresponding answers.

## Dataset Structure
The dataset contains:
- `prompt`: User question or query
- `text`: Additional reference information
- `context`: Relevant contextual background
- `answer`: Correct response to the question

## Project Components
- `data_preprocessing.py`: Prepares and formats the dataset for fine-tuning
- `model_finetuning.py`: Implements LoRA fine-tuning on Llama-3.2-3B-Instruct
- `evaluation.py`: Assesses model performance using BLEU, ROUGE, and Exact Match
- `utils.py`: Helper functions for data processing and model evaluation
- `config.py`: Configuration settings for the project

## Usage

### Environment Setup
```
pip install -r requirements.txt
```

### Data Preparation
```
python data_preprocessing.py --input_file your_dataset.csv --output_dir ./processed_data
```

### Fine-tuning
```
python model_finetuning.py --model_name meta-llama/Llama-3.2-3B-Instruct --data_dir ./processed_data --output_dir ./fine_tuned_model
```

### Evaluation
```
python evaluation.py --model_path ./fine_tuned_model --test_data ./processed_data/test.json
```

## Acceptance Criteria
- ✅ 10-15% improvement in BLEU, ROUGE, and Exact Match scores
- ✅ Correct utilization of contextual information
- ✅ 85%+ factually correct and contextually appropriate responses
- ✅ Ability to handle unseen queries within the domain
