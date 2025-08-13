"""
metrics.py

This module defines evaluation metrics for a joint Intent Classification and Slot Filling model.

Metrics Implemented:
    1. Intent Accuracy
    2. Slot F1 Score
    3. Joint Accuracy (both intent and slot predictions correct)
""" 
from seqeval.metrics import f1_score, classification_report
from sklearn.metrics import accuracy_score

def compute_intent_accuracy(pred_intents, true_intents):
    """
    Compute the accuracy of intent classification.

    Args: 
        pred_intents : list of str or list of int
            Predicted intent labels for each input example.
        true_intents : list of str or list of int
            Ground truth intent labels.

    Returns:
        float
            Intent classification accuracy (range: 0.0 - 1.0).
    """
    return accuracy_score(true_intents, pred_intents)

def compute_slot_f1(pred_slots, true_slots):
    """
    Compute the F1 score for slot filling.

    Args:
        pred_slots : list of list of str
            Predicted slot labels for each token in the sequence.
        true_slots : list of list of str
            Ground truth slot labels for each token in the sequence.

    Returns:
        float
            F1 score for slot predictions (range: 0.0 - 1.0).
    """
    return f1_score(true_slots, pred_slots)

def compute_joint_accuracy(pred_intents, true_intents, pred_slots, true_slots):
    """
    Compute the joint accuracy of intent classification and slot filling.

    Joint accuracy measures the proportion of examples where
    BOTH the predicted intent matches the true intent AND all slot labels match.

    Args:
        pred_intents : list of str or list of int
            Predicted intent labels.
        true_intents : list of str or list of int
            Ground truth intent labels.
        pred_slots : list of list of str
            Predicted slot labels for each token in the sequence.
        true_slots : list of list of str
            Ground truth slot labels for each token in the sequence.

    Returns:
        float
            Joint accuracy (range: 0.0 - 1.0).
    """
    joint_correct = 0
    for pi, ti, ps, ts in zip(pred_intents, true_intents, pred_slots, true_slots):
        if pi == ti and ps == ts:
            joint_correct += 1
    return joint_correct / len(true_intents)
