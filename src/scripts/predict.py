"""
This module provides functions to load a trained JointBERT model
and make predictions (intent classification and slot filling) 
for a given text input.

It includes:
1. `load_model`       - Loads a trained model from disk.
2. `predict_single`   - Makes predictions for a single input sequence.
"""
import torch

# def load_model(model_path, model_class, config, device):
#     """
#     Load a trained model from a given file path.

#     Args:
#         model_path (str): Path to the saved model state dictionary (.pt or .bin file).
#         model_class (torch.nn.Module): Model class to instantiate.
#         config (dict or object): Model configuration parameters.
#         device (str): Device to load the model on ("cuda" or "cpu").

#     Returns:
#         model (torch.nn.Module): Loaded and ready-to-use model.
#     """    
#     model = model_class(config) # Initialize model with given configuration
#     model.load_state_dict(torch.load(model_path, map_location=device)) # Load weights
#     model.to(device) # Move model to device
#     model.eval() # Set model to evaluation mode
#     return model


def predict_single(
        text, model, tokenizer, intent2id, slot2id, 
        max_len=64, device="cuda" if torch.cuda.is_available() else "cpu"
    ):
    """
    Perform intent classification and slot filling for a single text input.

    Args:
        text (list[str]): Tokenized input sequence (list of words).
        model (torch.nn.Module): Trained JointBERT model.
        tokenizer (TokenizerWrapper): Tokenizer wrapper for encoding input text.
        intent2id (dict): Mapping from intent labels to IDs.
        slot2id (dict): Mapping from slot labels to IDs.
        max_len (int, optional): Maximum sequence length for padding/truncation. Default is 64.
        device (str, optional): Device to perform inference on ("cuda" or "cpu").

    Returns:
        dict: {
            "intent": str,       # Predicted intent label
            "slots": list[tuple] # List of (word, slot_label) pairs
        }
    """
    # Create reverse mappings for IDs to labels
    id2intent = {v: k for k, v in intent2id.items()}
    id2slot = {v: k for k, v in slot2id.items()}

    # Tokenize input sequence
    # tok = tokenizer.tokenizer if hasattr(tokenizer, "tokenizer") else tokenizer
    encoding = tokenizer(
        text,
        is_split_into_words=True,
        return_offsets_mapping=True,
        padding='max_length',
        truncation=True,
        max_length=max_len,
        return_tensors='pt'
    )

    # Extract tokenized tensors
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    # Disable gradient computation for inference
    with torch.no_grad():
        intent_logits, slot_logits = model(input_ids, attention_mask)

    # Intent prediction
    intent_id = torch.argmax(intent_logits, dim=1).item()
    intent_label = id2intent[intent_id]

    # Slot prediction
    slot_ids = torch.argmax(slot_logits, dim=2).squeeze(0).tolist()
    # tokens = tokenizer.tokenizer.convert_ids_to_tokens(input_ids.squeeze(0))
    word_ids = encoding.word_ids(batch_index=0)

    # Align slot predictions with original words
    slot_labels = []
    previous_word_idx = None
    for idx, word_idx in enumerate(word_ids):
        if word_idx is None or word_idx == previous_word_idx:
            continue
        previous_word_idx = word_idx
        slot_labels.append(id2slot.get(slot_ids[idx], "O"))

    # Create slot output as (word, label) pairs
    words = text
    slot_output = list(zip(words, slot_labels))

    return {
        "intent": intent_label,
        "slots": slot_output
    }