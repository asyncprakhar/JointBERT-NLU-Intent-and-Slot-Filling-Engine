"""
Intent Classification & Slot Filling API
----------------------------------------
This FastAPI service loads a trained JointBERT model to perform:
  - Intent classification (predicting the overall purpose of the query)
  - Slot filling (extracting entities from the query)

Input:  Natural language sentence (string)
Output: Predicted intent and extracted slots

Example:
--------
POST /predict
{
    "sentence": "book a flight from new york to seattle"
}

Response:
{
    "intent": "book_flight",
    "slots": {
        "from_location": "new york",
        "to_location": "seattle"
    }
}
"""
import json
import os
import torch
import uvicorn
from fastapi import FastAPI, Query
from pydantic import BaseModel
from .model_initializer import JointBERT
from transformers import BertTokenizerFast

# ------------------------
# Step 1: Config paths
# ------------------------

# Defining model path
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ARTIFACTS_DIR = os.path.join(os.getcwd(), "artifacts")
SAVED_WEIGHTS_DIR = os.path.join(ARTIFACTS_DIR, "saved_weights")                      
MODEL_PATH = os.path.join(SAVED_WEIGHTS_DIR, "saved_model.pt") # directory containing your trained model files 
# Defining tokenizer path
TOKENIZER_PATH = os.path.join(ARTIFACTS_DIR, "tokenizer") # directory containing your trained tokenizer
# Defining intent and slot mapping paths
INTENT2ID_PATH = os.path.join(ARTIFACTS_DIR, "intent2id.json")
SLOT2ID_PATH = os.path.join(ARTIFACTS_DIR, "slot2id.json")

# ------------------------
# Step 2: Load artifacts (Tokenizer & Label Mappings)
# ------------------------

# Load the pre-trained tokenizer
tokenizer = BertTokenizerFast.from_pretrained(TOKENIZER_PATH)

# Load the intent and slot mappings
slot2id = json.load(open(SLOT2ID_PATH))
intent2id = json.load(open(INTENT2ID_PATH))

# Create reverse mappings for IDs to labels
id2intent = {v: k for k, v in intent2id.items()}
id2slot = {v: k for k, v in slot2id.items()}


# -------------------------------
# Step 3: Load Trained Model
# -------------------------------

# Initialize the model
model = JointBERT(
    num_intent_labels=len(intent2id),
    num_slot_labels=len(slot2id)
).to(DEVICE)

# Loading trained model weights
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)

# -------------------------------
# Step 4: FastAPI Setup
# -------------------------------

# Create FastAPI application instance
app = FastAPI(title="Intent & Slot Inference API")

# Request body schema
class QueryInput(BaseModel):
    sentence: str

# -------------------------------
# Step 5: Prediction Endpoint
# -------------------------------
@app.post("/predict")
def predict(data: QueryInput):
    """
    Predicts intent and slots from a natural language sentence.

    Steps:
    1. Convert the input sentence string into a list of tokens.
    2. Pass the tokens into the trained model for prediction.
    3. Return the predicted intent and slot mappings.
    """
    tokens = tokenizer.tokenize(data.sentence)
    inputs = tokenizer(
        tokens,
        is_split_into_words=True,
        return_tensors="pt",
        truncation=True,
        max_length=64,
        return_offsets_mapping=True,
        padding="max_length"
    )

    # Extract tokenized tensors
    input_ids = inputs['input_ids'].to(DEVICE)
    attention_mask = inputs['attention_mask'].to(DEVICE)

    # Disable gradient computation for inference
    with torch.no_grad():
        intent_logits, slot_logits = model(input_ids, attention_mask)

    # Intent prediction
    intent_id = torch.argmax(intent_logits, dim=1).item()
    intent_label = id2intent[intent_id]
    
    # Slot prediction
    slot_ids = torch.argmax(slot_logits, dim=2).squeeze(0).tolist()
    # tokens = tokenizer.tokenizer.convert_ids_to_tokens(input_ids.squeeze(0))
    word_ids = inputs.word_ids(batch_index=0)

    # Align slot predictions with original words
    slot_labels = []
    previous_word_idx = None
    for idx, word_idx in enumerate(word_ids):
        if word_idx is None or word_idx == previous_word_idx:
            continue
        previous_word_idx = word_idx
        slot_labels.append(id2slot.get(slot_ids[idx], "O"))

    # Create slot output as (word, label) pairs
    words = tokens
    slot_output = list(zip(words, slot_labels))

    # Return prediction results as JSON
    return {
        "intent": intent_label,
        "slots": slot_output
    }
# ------------------------
# Run the API
# ------------------------
if __name__ == "__main__":
    uvicorn.run("inference_api:app", host="0.0.0.0", port=8000, reload=True)
