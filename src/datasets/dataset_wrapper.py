'''
dataset_module.py
-----------------
PyTorch Dataset wrapper for joint intent classification and slot-filling tasks.
'''

import torch
from torch.utils.data import Dataset

class JointIntentSlotDataset(Dataset):
    '''
    PyTorch compatible dataset for joint intent classification and slot filling.

    Each item is a single utterance already tokenized and label-aligned by `TokenizerWrapper.tokenize_dataset()`.  
    The class turns that list of dictionaries into a Dataset object that can be fed directly to a `torch.utils.data.DataLoader`.

    Args:
        tokenized_data : list of dict
            Output of `TokenizerWrapper.tokenize_dataset()`.  
            Each dictionary contains:
                - "input_ids"     : torch.LongTensor shape (1, max_len)
                - "attention_mask": torch.LongTensor shape (1, max_len) 
                - "labels"        : list[int] length max_len (slot IDs,-100 on sub-tokens)
                - "intent_label"  : int (intent class ID)    
        
        Notes:
            The leading batch dimension produced by the tokenizer (size 1) is
            removed with tensor.squeeze(0) so every field ends up with shape
            (max_len,).
    '''
    def __init__(self, tokenized_data):
        # Strip the extra batch dimension from tokenizer outputs
        self.input_ids = [item["input_ids"].squeeze(0) for item in tokenized_data] 
        self.attention_mask = [item["attention_mask"].squeeze(0) for item in tokenized_data]
        # Convert labels to tensors for effortless stacking in DataLoader
        self.slot_labels = [torch.tensor(item["labels"]) for item in tokenized_data]
        self.intent_labels = [torch.tensor(item["intent_label"]) for item in tokenized_data]

    def __len__(self):
        '''Total number of examples'''
        return len(self.input_ids)

    def __getitem__(self, idx):
        '''
        Retrieves one example for DataLoader.
        
        Returns:
            dict
                {
                    "input_ids"     : torch.LongTensor shape (max_len,),
                    "attention_mask": torch.LongTensor shape (max_len,),
                    "slot_labels"   : torch.LongTensor shape (max_len,),
                    "intent_label"  : torch.LongTensor scalar
                }

        '''
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "slot_labels": self.slot_labels[idx],
            "intent_label": self.intent_labels[idx]
        }
