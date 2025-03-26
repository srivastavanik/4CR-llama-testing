#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Data preprocessing for LLM fine-tuning."""

import os
import json
import argparse
import pandas as pd
from typing import Dict, List, Tuple
from sklearn.model_selection import train_test_split
from config import Config

def load_data(file_path: str) -> pd.DataFrame:
    """Load data from csv file."""
    if file_path.endswith('.csv'):
        return pd.read_csv(file_path)
    elif file_path.endswith('.json'):
        return pd.read_json(file_path, lines=True)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess data for fine-tuning."""
    # Check if all required columns exist
    required_columns = ["prompt", "context", "answer"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Drop rows with missing values in required columns
    df = df.dropna(subset=required_columns)
    
    # Handle the context column - if it contains list-like strings, parse them
    if df["context"].dtype == object and df["context"].iloc[0].startswith("["):
        import ast
        df["context"] = df["context"].apply(lambda x: 
            "\n\n".join(ast.literal_eval(x)) if isinstance(x, str) and x.startswith("[") 
            else x
        )
    
    # Combine text and context if both exist
    if "text" in df.columns:
        df["context"] = df.apply(
            lambda row: f"{row['context']}\n\nAdditional information: {row['text']}"
            if not pd.isna(row.get('text')) else row['context'], 
            axis=1
        )
    
    # Format data for the model
    df["input_text"] = df.apply(
        lambda row: Config.PROMPT_TEMPLATE.format(
            context=row["context"],
            prompt=row["prompt"]
        ),
        axis=1
    )
    
    return df[['input_text', 'answer', 'prompt', 'context']]

def split_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data into train, validation and test sets."""
    # First split off the test set
    train_val_df, test_df = train_test_split(
        df, test_size=Config.TRAIN_TEST_SPLIT, random_state=Config.SEED
    )
    
    # Then split the train_val into train and validation
    train_df, val_df = train_test_split(
        train_val_df, 
        test_size=Config.VALIDATION_SPLIT, 
        random_state=Config.SEED
    )
    
    return train_df, val_df, test_df

def save_data(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, output_dir: str):
    """Save preprocessed data to output directory."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert DataFrames to lists of dictionaries
    train_data = train_df.to_dict(orient='records')
    val_data = val_df.to_dict(orient='records')
    test_data = test_df.to_dict(orient='records')
    
    # Save to JSON files
    with open(os.path.join(output_dir, 'train.json'), 'w') as f:
        json.dump(train_data, f, indent=2)
    
    with open(os.path.join(output_dir, 'validation.json'), 'w') as f:
        json.dump(val_data, f, indent=2)
    
    with open(os.path.join(output_dir, 'test.json'), 'w') as f:
        json.dump(test_data, f, indent=2)
    
    print(f"Saved preprocessed data to {output_dir}")
    print(f"Train set size: {len(train_data)}")
    print(f"Validation set size: {len(val_data)}")
    print(f"Test set size: {len(test_data)}")

def main():
    parser = argparse.ArgumentParser(description="Preprocess data for LLM fine-tuning")
    parser.add_argument(
        "--input_file", 
        type=str, 
        required=True, 
        help="Path to input data file (CSV or JSON)"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default=Config.PROCESSED_DATA_DIR,
        help="Directory to save preprocessed data"
    )
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from {args.input_file}...")
    df = load_data(args.input_file)
    
    # Preprocess data
    print("Preprocessing data...")
    preprocessed_df = preprocess_data(df)
    
    # Split data
    print("Splitting data into train, validation, and test sets...")
    train_df, val_df, test_df = split_data(preprocessed_df)
    
    # Save data
    save_data(train_df, val_df, test_df, args.output_dir)
    
if __name__ == "__main__":
    main()
