"""
evaluate.py

This script evaluates a trained Joint Intent Classification & Slot Filling model (JointBERT)
on a given dataset and returns key performance metrics.

It uses:
- Accuracy for intent prediction
- F1 score for slot tagging
- Joint accuracy (both intent and slots must be correct for a sample)
"""
import torch
from tqdm import tqdm

# Custom metric functions (defined in utils/metrics.py)
from src.utils import (
    compute_intent_accuracy,
    compute_slot_f1,
    compute_joint_accuracy
)


def evaluate_model(model, dataloader, id2slot=None, device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Evaluates the model on a given dataset.

    Args:
        model (torch.nn.Module):
            Trained JointBERT model for intent classification and slot filling.

        dataloader (torch.utils.data.DataLoader):
            DataLoader that provides batches of evaluation data.
            Each batch must be a dictionary containing:
                - "input_ids": token IDs for each input sentence
                - "attention_mask": attention mask for padded tokens
                - "intent_label": integer label for the intent
                - "slot_labels": integer labels for slot tags, with -100 for ignored positions

        id2slot (dict, optional):
            Mapping from slot label IDs to slot label strings (e.g., {0: "O", 1: "B-LOC", ...}).
            Needed for converting numeric predictions to readable labels for F1 computation.
            If None, numeric IDs will be used directly.

        device (str, optional):
            Device to run evaluation on ("cuda" or "cpu").
            Defaults to GPU if available, otherwise CPU.

    Returns:
        dict: A dictionary containing:
            - "intent_accuracy" (float): Accuracy score for intent classification
            - "slot_f1" (float): F1 score for slot tagging
            - "joint_accuracy" (float): Percentage of samples where BOTH
              intent and slot predictions are fully correct

    Notes:
        - `torch.no_grad()` is used to avoid storing gradients during evaluation, saving memory.
        - Slot labels with ID = -100 are ignored (these are padding tokens).
        - Joint accuracy is stricter than intent or slot accuracy individually.
    """
    
    # Send model to device and set it to evaluation mode
    model.to(device)
    model.eval()

    # Store predictions and true labels for metrics
    all_intent_preds = []
    all_intent_labels = []

    all_slot_preds = []
    all_slot_labels = []

    # Disable gradient computation for evaluation
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="🔍 Evaluating"):

            # Move batch tensors to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            intent_ids = batch["intent_label"].to(device)
            slot_ids = batch["slot_labels"].to(device)

            # Forward pass → get logits (raw predictions before softmax)
            intent_logits, slot_logits = model(input_ids, attention_mask)

            # ===== Intent Prediction =====
            # Select class with highest probability for each sample
            intent_preds = torch.argmax(intent_logits, dim=1)
            all_intent_preds.extend(intent_preds.cpu().tolist())
            all_intent_labels.extend(intent_ids.cpu().tolist())

            # ===== Slot Prediction =====
            # Select highest probability tag for each token
            slot_preds = torch.argmax(slot_logits, dim=2)
            
            # Process each sequence in batch
            for i in range(input_ids.size(0)): # Loop over sentences
                true_seq = []
                pred_seq = []
                for j in range(input_ids.size(1)): # Loop over tokens
                    # Ignore padding tokens
                    if slot_ids[i][j] != -100:
                        label_id = slot_ids[i][j].item()
                        pred_id = slot_preds[i][j].item()
                        true_seq.append(id2slot[label_id] if id2slot else label_id)
                        pred_seq.append(id2slot[pred_id] if id2slot else pred_id)
                        
                # Append full sentence predictions
                all_slot_labels.append(true_seq)
                all_slot_preds.append(pred_seq)

    # ===== Compute Evaluation Metrics =====
    intent_acc = compute_intent_accuracy(all_intent_preds, all_intent_labels)
    slot_f1 = compute_slot_f1(all_slot_preds, all_slot_labels)
    joint_acc = compute_joint_accuracy(all_intent_preds, all_intent_labels, all_slot_preds, all_slot_labels)

    # ===== Print Results ===== 
    print("\n📊 ======= Evaluation Results =======")
    print(f"Intent Accuracy : {intent_acc:.4f}")
    print(f"Slot F1 Score   : {slot_f1:.4f}")
    print(f"Joint Accuracy  : {joint_acc:.4f}")
    print("====================================\n")

    return {
        "intent_accuracy": intent_acc,
        "slot_f1": slot_f1,
        "joint_accuracy": joint_acc
    }
