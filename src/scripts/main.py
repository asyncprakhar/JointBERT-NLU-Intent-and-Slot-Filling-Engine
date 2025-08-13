"""
Main entry point for the Joint Intent & Slot Classifier project.

This script provides a CLI interface to:
1. Train a model (`train` mode)
2. Evaluate a trained model (`eval` mode)
3. Make predictions on a sample input (`predict` mode)

It integrates all the modules:
- Tokenizer wrapper
- Dataset preparation
- Model definition (JointBERT)
- Training, evaluation, and prediction utilities

Dataset format:
    Input datasets must be in JSONL format where each line is:
        {"tokens": ["word1", "word2", ...],
         "slots": ["O", "B-LOC", ...],
         "intent": "intent_label"}

Usage:
    Train:
        python main.py train
    Evaluate:
        python main.py eval
    Predict:
        python main.py predict
"""

import os
import argparse
import torch
import gc
import sys
import traceback

from torch.utils.data import DataLoader
from src.utils import (TokenizerWrapper, save_artifacts, artifacts_loader)
from src.datasets.dataset_wrapper import JointIntentSlotDataset
from src.models import (JointBERT, JointLoss)
from src.scripts.train import train_model
from src.scripts.evaluate import evaluate_model
from src.scripts.predict import predict_single

# ---------------------------
# Global settings & constants
# ---------------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ARTIFACTS_DIR = os.path.join(os.getcwd(), "artifacts")              # Define main artifacts directory
SAVED_WEIGHTS_DIR = os.path.join(ARTIFACTS_DIR, "saved_weights")    # Define subdirectory for saved weights
os.makedirs(SAVED_WEIGHTS_DIR, exist_ok=True)                       # Ensure directories exist
MODEL_PATH = os.path.join(SAVED_WEIGHTS_DIR, "saved_model.pt")      # Define full model save path
TOKENIZER_NAME = "bert-base-uncased"                                # Pretrained BERT tokenizer
MAX_LEN = 64                                                        # Max sequence length for tokenization
BATCH_SIZE = 8                                                      # Batch size for DataLoader
LEARNING_RATE = 2e-5 
EPOCHS = 5

# ---------------------------
# Utility functions
# ---------------------------

def clear_memory():
    """
    Free unused CPU/GPU memory.

    This function:
    - Runs Python garbage collection (`gc.collect`)
    - Clears GPU cache if using CUDA
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def load_tokenizer_and_data():
    """
    Load the tokenizer and prepare datasets for training & validation.

    Returns:
        tokenizer (TokenizerWrapper): Pretrained BERT tokenizer wrapper.
        train_dataset (Dataset): Encoded training dataset.
        val_dataset (Dataset): Encoded validation dataset.
        slot2id (dict): Slot label → integer ID mapping.
        intent2id (dict): Intent label → integer ID mapping.

    Raises:
        FileNotFoundError: If JSONL dataset files are missing.
    """
    tokenizer = TokenizerWrapper(TOKENIZER_NAME, max_len=MAX_LEN)

    # Load JSONL datasets
    train_data = tokenizer.load_jsonl("data/atis_jsonl/atis_train_dataset.jsonl")
    val_data = tokenizer.load_jsonl("data/atis_jsonl/atis_val_dataset.jsonl")

    # Create label mappings
    slot2id, intent2id = tokenizer.create_label_mappings(train_data)

    # Tokenize datasets
    train_encodings = tokenizer.tokenize_dataset(train_data)
    val_encodings = tokenizer.tokenize_dataset(val_data)

    # Create dataset objects
    train_dataset = JointIntentSlotDataset(train_encodings)
    val_dataset = JointIntentSlotDataset(val_encodings)

    return tokenizer, train_dataset, val_dataset, slot2id, intent2id

# ---------------------------
# Mode functions
# ---------------------------
def run_train():
    """
    Train the JointBERT model.

    Steps:
    - Load tokenizer and training dataset.
    - Initialize JointBERT model with correct label sizes.
    - Train for a fixed number of epochs.
    - Save model weights and tokenizer artifacts to disk.

    Side effects:
        Creates 'artifacts/' folder containing:
        - saved_model.pt (model weights)
        - label mappings
        - tokenizer files
    """
    try:
        # Load tokenizer and datasets
        tokenizer, train_dataset, _, slot2id, intent2id = load_tokenizer_and_data()

        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        
        # Initialize model
        model = JointBERT(
            num_intent_labels=len(intent2id),
            num_slot_labels=len(slot2id)
        ).to(DEVICE)

        # Define loss and optimizer
        criterion = JointLoss()
        optimizer = torch.optim.AdamW(model.parameters(), LEARNING_RATE)

        # Train the model
        train_model(model, train_loader, criterion, optimizer, DEVICE, EPOCHS)

        # Save the artifacts and model
        save_artifacts(tokenizer, intent2id, slot2id, ARTIFACTS_DIR)
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"\n✅ Model saved to: {MODEL_PATH}")

    except Exception as e:
        print(f"\n❌ Training failed:\n{traceback.format_exc()}")
    finally:
        for var in ["model", "train_loader", "tokenizer"]:
            if var in locals():
                del locals()[var]
        clear_memory()

def run_eval():
    """
    Evaluate the trained model on the test dataset.

    Steps:
    - Load tokenizer, slot & intent mappings.
    - Tokenize test dataset.
    - Load trained model weights.
    - Run evaluation metrics on test set.
    """
    try:
        # loading tokenizer, slot2id and intent2id
        tokenizer, slot2id, intent2id = artifacts_loader(ARTIFACTS_DIR)

        # laoding, tokenizing, wrapping and passing the data in the torch.utils.data.Dataloader
        test_data = tokenizer.load_jsonl("data/train_filtered.jsonl") # load the data from the jsonl file
        test_encodings = tokenizer.tokenize_dataset(test_data) # tokenize the data
        test_dataset = JointIntentSlotDataset(test_encodings) # wrapping the data under the torch.tensor to pass it into torch.utils.data.DataLoader
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE) # creating the data loader

        # Load model
        model = JointBERT(
            num_intent_labels=len(intent2id),
            num_slot_labels=len(slot2id)
        ).to(DEVICE)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))

        # Create reverse mapping for slot labels
        id2slot = {i: label for label, i in slot2id.items()}

        # Run evaluation
        evaluate_model(model, test_loader, id2slot, DEVICE)

    except Exception as e:
        print(f"\n❌ Evaluation failed:\n{traceback.format_exc()}")
    finally:
        for var in ["model", "test_loader", "tokenizer"]:
            if var in locals():
                del locals()[var]
        clear_memory()

def run_predict(sentence: str ):
    """
    Make predictions on a single example.

    Steps:
    - Load tokenizer, slot & intent mappings.
    - Load trained model.
    - Run prediction and print:
        - Tokens
        - Predicted intent
        - Predicted slots for each token
    """
    try:
        # loading tokenizer, slot2id and intent2id
        tokenizer, slot2id, intent2id = artifacts_loader(ARTIFACTS_DIR)

        # The sentence is now an argument, split it into tokens
        tokens = sentence.split()
        # Load model
        model = JointBERT(
            num_intent_labels=len(intent2id),
            num_slot_labels=len(slot2id)
        ).to(DEVICE)

        # Make prediction
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        result = predict_single(tokens, model, tokenizer, intent2id, slot2id, MAX_LEN, DEVICE)

        # Display results
        print("\n🔍 Prediction")
        print(f"Tokens: {tokens}")
        print(f"Intent: {result['intent']}")
        print(f"Slots : {result['slots']}")

    except Exception as e:
        print(f"This is error: {e}")
        print(f"\n❌ Prediction failed:\n{traceback.format_exc()}")
    finally:
        for var in ["model", "tokenizer"]:
            if var in locals():
                del locals()[var]
        clear_memory()

# ---------------------------
# Main entry point
# ---------------------------

def main():
    """
    CLI parser that decides whether to train, evaluate, or predict 
    based on the provided command-line argument.
    
    parser chooses between:
        'train' → run_train()
        'eval' → run_eval()
        'predict' → run_predict()
    """
    parser = argparse.ArgumentParser(description="Joint Intent & Slot Classifier CLI")
    parser.add_argument("mode", choices=["train", "eval", "predict"], help="Choose an action to perform")
    parser.add_argument("--sentence", type=str, help="The sentence to predict on (required for 'predict' mode).")
    
    args = parser.parse_args()

    if args.mode == "train":
        run_train()
    elif args.mode == "eval":
        run_eval()
    elif args.mode == "predict":
        if not args.sentence:
            # If predict mode is chosen but no sentence is given, print error and exit.
            print("❌ Error: The 'predict' mode requires a --sentence argument.", file=sys.stderr)
            parser.print_help()
            sys.exit(1)
        run_predict(args.sentence)
if __name__ == "__main__":
    main()
