import torch.nn as nn
from transformers import BertModel

class JointBERT(nn.Module):
    '''
    BERT backbone with two specific heads

    Args:
        bert_model_name: str, default="bert-base-uncased"
            Hugging Face model identifier for the pretrained BERT encoder.
        num_intent_classes: int, default=10
            Number of distinct intent classes.
        num_slot_labels : int, default=20
           Number of distinct slot labels (including O).

    Attributes:
        bert : transformers.BertModel
            Shared contextual encoder.
        intent_classifier : torch.nn.Linear
            Maps pooled CLS embedding → intent logits
            (shape (batch, num_intent_labels)).
        slot_classifier : torch.nn.Linear
            Maps each token embedding → slot logits
            (shape (batch, seq_len, num_slot_labels)).
    
    '''
    def __init__(self, 
                 bert_model_name="bert-base-uncased", 
                 num_intent_labels=10, 
                 num_slot_labels=20):
        super(JointBERT, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name) # load pre-trained BERT model

        # Intent classification head
        self.intent_classifier = nn.Linear(self.bert.config.hidden_size, num_intent_labels)

        # Slot filling head
        self.slot_classifier = nn.Linear(self.bert.config.hidden_size, num_slot_labels)

    def forward(self, input_ids, attention_mask):
        '''
        Forward pass.

        Args:
            input_ids: torch.LongTensor
                Shape (batchsize, seq_len) - input token ids.
            attention_mask : torch.LongTensor
                Shape (batch_size, seq_len) - 1 for real tokens, 0 for padding.

        Returns:
            intent_logits : torch.FloatTensor
                Shape (batch_size, num_intent_labels) - raw scores.
            slot_logits : torch.FloatTensor
                Shape (batch_size, seq_len, num_slot_labels) - raw scores 
        '''
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        sequence_output = outputs.last_hidden_state  # (batch_size, seq_len, hidden)
        pooled_output = outputs.pooler_output        # (batch_size, hidden)

        # Classification heads
        intent_logits = self.intent_classifier(pooled_output)     # intent prediction
        slot_logits = self.slot_classifier(sequence_output)       # slot prediction

        return intent_logits, slot_logits
