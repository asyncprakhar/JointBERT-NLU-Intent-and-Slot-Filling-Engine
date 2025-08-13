# tokenizer_module.py

import json
from transformers import BertTokenizerFast

class TokenizerWrapper:
    def __init__(self, tokenizer_name="bert-base-uncased", max_len=128):
        self.tokenizer = BertTokenizerFast.from_pretrained(tokenizer_name)
        self.max_len = max_len
        self.slot2id = {}
        self.intent2id = {}

    # 1. Load JSONL data
    @staticmethod
    def load_jsonl(path):
        '''
            Reads a JSON Lines (jsonl) file.
            
            Args: 
                path (str): Path of the jsonl file.

            Returns: 
                list of dict: List where each dict contains keys "tokens", "slots", and "intent" as per dataset specification.
        '''
        with open(path, 'r', encoding='utf-8') as f:
            return [json.loads(line.strip()) for line in f]

    # 2. Create label mappings
    def create_label_mappings(self, dataset):
        '''
        Creates mappings from slot labels and intent labels to unique integer IDs.
        
        Args: 
            dataset (list of dict): 
                List of dataset examples
                Each example is a dict with fields:
                - "tokens": list of str
                - "slots": list of str
                - "intent": str
                    e.g., {"tokens": ["which", "flights", "travel", "from", "nashville", "to", "tacoma"], 
                        "slots": ["O", "O", "O", "O", "B-fromloc.city_name", "O", "B-toloc.city_name"], 
                        "intent": "atis_flight"
                        }

        Returns: 
            slot2id (dict): 
                Mapping from slot label string to unique int ID.
                    e.g., {'B-aircraft_code': 0 , 'B-airline_code': 1 , 'B-airline_name': 2 , ...., 'O': 95}
            intent2id (dict):
                Mapping from intent string to unique int ID.
                    e.g., {'atis_abbreviation': 0, 'atis_aircraft': 1, 'atis_airfare': 2, ...., 'atis_restriction': 15}                
        '''
        slot_labels = set()
        intent_labels = set()

        for example in dataset:
            slot_labels.update(example["slots"])
            intent_labels.add(example["intent"])

        self.slot2id = {label: i for i, label in enumerate(sorted(slot_labels))}
        self.intent2id = {label: i for i, label in enumerate(sorted(intent_labels))}
        # self.id2slot = {i: label for label, i in self.slot2id.items()}
        # self.id2intent = {i: label for label, i in self.intent2id.items()}

        return self.slot2id, self.intent2id

    # 3. Tokenize and align slot labels
    def tokenize_and_align(self, example):
        '''
        Tokenizes an input example and aligns slot labels with the tokenized representation.

        Args : 
            example (dict): 
                A single entry in the dataset, with keys:
                    - "tokens": list of str, e.g., ["which", "flights", ...]
                    - "slots": list of str, e.g., ["O", "O", ...]
                    - "intent": str, e.g., "atis_flight" 

                e.g. {"tokens": ["which", "flights", "travel", "from", "nashville", "to", "tacoma"], 
                       "slots": ["O", "O", "O", "O", "B-fromloc.city_name", "O", "B-toloc.city_name"], 
                       "intent": "atis_flight"
                    }

        Returns : 
            encodings (dict): 
                Dictionary with the following entries:
                    - "input_ids": torch.LongTensor of shape (1, max_length) 
                        IDs of input tokens (padded).
                    - "token_type_ids": torch.LongTensor of shape (1, max_length) 
                        Segment IDs (all zeros for single sentence inputs). 
                    - "attention_mask": torch.LongTensor of shape (1, max_length)
                        1 for actual token, 0 for paddeding.
                    - "offset_mapping": torch.LongTensor of shape (1, max_length, 2)
                        Character span for each token.
                    - "labels": list of int of max_length
                        Aligned slot labels: slot index if token is first subword of a word, -100 otherwise (to be ignored by loss functions).
                    - "intent_label": int
                        Index of intent label from intent2id.
            Note: encoding is HuggingFace tokenizer output with added keys "labels" and "intent_label".
        '''
        encoding = self.tokenizer(
            example["tokens"],
            is_split_into_words=True,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_offsets_mapping=True,
            return_tensors="pt"
        )

        word_ids = encoding.word_ids(batch_index=0)
        previous_word_idx = None
        aligned_slots = []

        for word_idx in word_ids:
            if word_idx is None:
                aligned_slots.append(-100)
            elif word_idx != previous_word_idx:
                aligned_slots.append(self.slot2id[example["slots"][word_idx]])
            else:
                aligned_slots.append(-100)
            previous_word_idx = word_idx

        encoding["labels"] = aligned_slots
        encoding["intent_label"] = self.intent2id[example["intent"]]
        return encoding

    #4. Tokenize full dataset
    def tokenize_dataset(self, dataset):
        '''
        Tokenizes a list of dataset examples and aligns their slot and intent labels.

        Args:
            dataset (list of dict): 
                List of examples, where each example is a dictionary with the following keys:
                    - "tokens": list of str, e.g., ["which", "flights", ...]
                    - "slots": list of str, e.g., ["O", "O", ...]
                    - "intent": str, e.g., "atis_flight" 
                    
        Returns:
            encodings (list of dict): 
                List of encodings, one per input example. 
                Each encoding is a dict with the following keys:
                    - 'input_ids' (torch.LongTensor): shape (1, max_length)
                    - 'token_type_ids' (torch.LongTensor): shape (1, max_length)
                    - 'attention_mask' (torch.LongTensor): shape (1, max_length)
                    - 'offset_mapping' (torch.LongTensor): shape (1, max_length, 2)
                    - 'labels' (list of int): length max_length, aligned slot labels
                    - 'intent_label' (int): intent label id
        '''
        return [self.tokenize_and_align(example) for example in dataset]
    
    def save_pretrained(self, save_directory):
        self.tokenizer.save_pretrained(save_directory)

