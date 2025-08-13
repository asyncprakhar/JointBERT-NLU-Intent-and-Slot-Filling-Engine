"""
loss.py
---------------
Custom joint loss for intent classification + slot filling.
"""
import torch.nn as nn

class JointLoss(nn.Module):
    '''
    Weighted sum of intent-level and slot-level cross-entropy losses.

    Args:
        intent_loss_weight : float, default=1.0
            Multiplier for the intent loss term.
        slot_loss_weight : float, default=1.0
            Multiplier for the slot loss term.
        ignore_index : int, default=-100
            Label value to ignore in slot loss (used for sub-tokens).
    
    Note:
        * Intent loss uses `nn.CrossEntropyLoss` over the CLS-based logits.
        * Slot loss flattens `slot_logits` to shape `(batch*seq_len, num_slots)`.
        * `intent_loss_weight` and `slot_loss_weight` are used to change training emphasis between the two tasks.
    '''
    def __init__(self, intent_loss_weight=1.0, slot_loss_weight=1.0, ignore_index=-100):
        super(JointLoss, self).__init__()
        self.intent_loss_fn = nn.CrossEntropyLoss()
        self.slot_loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.intent_loss_weight = intent_loss_weight
        self.slot_loss_weight = slot_loss_weight

    def forward(self, intent_logits, slot_logits, intent_labels, slot_labels):
        '''
        Compute joint loss.

        Args:
            intent_logits : torch.FloatTensor
                Shape `(batch_size, num_intent_labels)`.
            slot_logits : torch.FloatTensor
                Shape `(batch_size, seq_len, num_slot_labels)`.
            intent_labels : torch.LongTensor
                Shape `(batch_size,)` - gold intent IDs.
            slot_labels : torch.LongTensor
                Shape `(batch_size, seq_len)` - gold slot IDs.

        Returns: 
            torch.Tensor
                Scalar total loss = intent_loss_weight * CE_intent + slot_loss_weight * CE_slot.
        '''
        
        intent_loss = self.intent_loss_fn(intent_logits, intent_labels)
        slot_loss = self.slot_loss_fn(slot_logits.view(-1, slot_logits.shape[-1]), slot_labels.view(-1))
        total_loss = self.intent_loss_weight * intent_loss + self.slot_loss_weight * slot_loss
        return total_loss
