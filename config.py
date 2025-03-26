"""Configuration for LLM fine-tuning."""

import torch

class Config:
    # Model settings
    MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
    MAX_LENGTH = 1024  # Reduced to save memory
    MAX_TARGET_LENGTH = 256  # Reduced to save memory
    PROMPT_TEMPLATE = """{prompt}

Context: {context}"""
    
    # Training settings
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 0.01
    NUM_EPOCHS = 1  # Keep at 1 for initial run
    BATCH_SIZE = 4  # Increased for GPU training
    GRADIENT_ACCUMULATION_STEPS = 2  # Reduced for GPU training
    WARMUP_STEPS = 50
    LOGGING_STEPS = 5
    SAVE_STRATEGY = "epoch"
    
    # LoRA settings
    LORA_R = 8  # Reduced rank for faster training
    LORA_ALPHA = 16
    LORA_DROPOUT = 0.05
    LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]
    
    # Data settings
    TRAIN_TEST_SPLIT = 0.2  # Increased to use less training data
    VALIDATION_SPLIT = 0.2  # Increased to use less validation data
    SEED = 42
    
    # Paths
    DATA_DIR = "./data"
    OUTPUT_DIR = "./fine_tuned_model"
    PROCESSED_DATA_DIR = "./processed_data"
    
    # Evaluation settings
    EVAL_BATCH_SIZE = 4  # Increased for GPU evaluation
    EVAL_METRICS = ["bleu", "rouge", "exact_match"]
    
    # Device settings
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # Use GPU if available
