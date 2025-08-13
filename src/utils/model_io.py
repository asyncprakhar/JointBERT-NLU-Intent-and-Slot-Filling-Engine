'''
model_io.py
-----------------
Utility functions for saving and loading model-related artifacts
(such as tokenizer and label mappings) for an intent classification
and slot filling task.

These functions ensure that:
- The trained tokenizer is stored in a reusable format
- Label-to-ID mappings for intents and slots are saved for consistent inference
- Artifacts can be reloaded for prediction without retraining
'''

import json
import os
from transformers import BertTokenizerFast
from pathlib import Path

def save_artifacts(tokenizer, intent2id, slot2id, save_dir):
    '''
    Save model artifacts (tokenizer, intent label mapping, slot label mapping) to disk.

    Args:
        tokenizer (transformers.BertTokenizerFast):
            The tokenizer used for tokenizing input text.
            This should be the same tokenizer used during training.
        
        intent2id (dict):
            A mapping from intent label names (str) to their corresponding integer IDs (int).
            Example:
                {
                    "atis_flight": 0,
                    "atis_airfare": 1
                }
        
        slot2id (dict):
            A mapping from slot label names (str) to their corresponding integer IDs (int).
            Example:
                {
                    "O": 0,
                    "B-fromloc.city_name": 1
                }
        
        save_dir (str):
            Path to the directory where artifacts will be saved.
            If it doesn't exist, it will be created automatically.

    Saves:
        - Tokenizer files in: `<save_dir>/tokenizer`
        - Intent mapping file: `<save_dir>/intent2id.json`
        - Slot mapping file: `<save_dir>/slot2id.json`
    '''
    os.makedirs(save_dir, exist_ok=True)

    # Save tokenizer in HuggingFace format
    tokenizer.save_pretrained(os.path.join(save_dir, "tokenizer"))

    # Save intent mapping as JSON
    with open(os.path.join(save_dir, "intent2id.json"), "w") as f:
        json.dump(intent2id, f, indent=4)

    # Save slot mapping as JSON
    with open(os.path.join(save_dir, "slot2id.json"), "w") as f:
        json.dump(slot2id, f, indent=4)

    print(f"[INFO] Artifacts saved to {save_dir}")

def artifacts_loader(artifacts_dir):
    '''
    Load tokenizer, slot label mapping, and intent label mapping from disk.

    Args:
        artifacts_dir (str or Path):
            Path to the directory containing:
                - `tokenizer/` folder
                - `slot2id.json`
                - `intent2id.json`

    Returns:
        tuple:
            tokenizer (transformers.BertTokenizerFast)
            slot2id (dict): Mapping from slot labels to IDs
            intent2id (dict): Mapping from intent labels to IDs

    Raises:
        ValueError:
            If the artifacts directory does not exist.
        
        FileNotFoundError:
            If any required file (tokenizer, slot2id.json, intent2id.json) is missing.

    Example:
        >>> tokenizer, slot2id, intent2id = artifacts_loader("saved_artifacts/")
    '''

    artifacts_dir = Path(artifacts_dir)

    if not artifacts_dir.exists():
        raise ValueError(f"Artifacts directory {artifacts_dir} does not exist")
    
    # Load tokenizer
    tokenizer_path = artifacts_dir / "tokenizer"
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer folder not found at {tokenizer_path}")
    tokenizer = BertTokenizerFast.from_pretrained(str(tokenizer_path))

    # Load slot2id mapping
    slot2id_path = artifacts_dir / "slot2id.json"
    if not slot2id_path.exists():
        raise FileNotFoundError(f"slot2id.json not found at {slot2id_path}")
    with open(slot2id_path, "r") as f:
        slot2id = json.load(f)

    # Load intent2id mapping
    intent2id_path = artifacts_dir / "intent2id.json"
    if not intent2id_path.exists():
        raise FileNotFoundError(f"intent2id.json not found at {intent2id_path}")
    with open(intent2id_path, "r") as f:
        intent2id = json.load(f)

    return tokenizer, slot2id, intent2id
