## JointBERT NLU: Intent and Slot Filling Engine
### Overview

JointBERT NLU is a powerful Natural Language Understanding (NLU) engine built using PyTorch and Transformers. It performs Joint Intent Recognition and Slot Filling, which means it can simultaneously identify a user's intent and extract relevant information (slots) from a sentence. This is a crucial component for building conversational AI, chatbots, and voice assistants.

This project provides a complete workflow, including training, evaluation, and deployment via a production-ready FastAPI service.


## Tech Stack ⚙️

- Python 3.8+

- PyTorch: For building and training the deep learning model.

- Transformers (Hugging Face): For leveraging the pre-trained BERT model.

- FastAPI: For serving the model via a high-performance REST API.

- Uvicorn: As the ASGI server for FastAPI.

- Seqeval: For F1-score metric calculation for slot filling.

## Project Structure
The repository is organized to separate data, source code, and model artifacts clearly.

```
project/
│
├── artifacts/                # Saved model weights, tokenizer, and label maps for inference
│   ├── saved_weights/
│   │   └── saved_model.pt
│   ├── tokenizer/
│   └── ...
│
├── data/                     # Raw and preprocessed datasets (ATIS, SNIPS)
│   ├── atis/
│   ├── snips/
│   └── ...
│
├── src/                      # All Python source code
│   ├── api/                  # FastAPI application for serving the model
│   │   ├── inference.py      # API endpoint logic
│   │   └── model_initializer.py
│   │
│   ├── datasets/             # PyTorch Dataset wrapper
│   │   └── dataset_wrapper.py
│   │
│   ├── models/               # Model architecture (JointBERT) and loss function
│   │   ├── model_module.py
│   │   └── loss.py
│   │
│   ├── scripts/              # Command-line interface scripts
│   │   ├── main.py           # Main CLI entrypoint (train, eval, predict)
│   │   ├── train.py          # Training loop
│   │   ├── evaluate.py       # Evaluation logic
│   │   └── predict.py        # Single prediction logic
│   │
│   └── utils/                # Utility functions for tokenization, I/O, and metrics
│       ├── tokenizer_module.py
│       ├── model_io.py
│       └── metrics.py
│
├── requirements.txt          # Project dependencies
└── README.md                 # You are here!
```

## Key Features ✨
**Joint Model Architecture:** A single BERT-based model with two output heads for efficient and effective intent and slot prediction.

**Training Pipeline:** A complete script to train the model from scratch on provided datasets.

**Model Evaluation:** Robust evaluation script reporting key metrics like Intent Accuracy, Slot F1-score, and Joint Accuracy.

**CLI Prediction:** Instantly test the model on a single sentence directly from your terminal.

**REST API:** A ready-to-use FastAPI endpoint to serve predictions over HTTP.

## Getting Started
Follow these steps to set up and run the project locally.

### Prerequisites
- Python 3.8 or higher

- pip package installer

### Installation
1. Clone the repository:
```
git clone https://github.com/your-username/JointBERT-NLU.git
cd JointBERT-NLU
```

2. Create and activate a virtual environment:

- On macOS/Linux:

```python3 -m venv venv
source venv/bin/activate
```

On Windows:
```
python -m venv venv
.\venv\Scripts\activate
```
3. Install the required dependencies:
```
pip install -r requirements.txt
```
4. Usage

The project can be operated in several modes via the main CLI script src/scripts/main.py.

5. Training

To train the model, run the train command. This will process the dataset, train the JointBERT model, and save all necessary assets (model weights, tokenizer, label maps) to the artifacts/ directory.

```
python src/scripts/main.py train --dataset atis --model_name_or_path bert-base-uncased --epochs 5 --batch_size 32
```

6. Evaluation

To evaluate a trained model, use the eval command. This will load the artifacts and compute metrics on the test set.
```
python src/scripts/main.py eval --dataset atis
```

This will report the following metrics:

Intent Accuracy

Slot F1-Score

Joint Accuracy (exact match for both intent and all slots)

Prediction (CLI)
To test the model with a single sentence, use the predict command.

```
python src/scripts/main.py predict --sentence "show me flights from new york to san francisco"
```
Running the API
To start the inference server, run the FastAPI application using uvicorn.

Bash

uvicorn src.api.inference:app --host 0.0.0.0 --port 8000 --reload
The API will be available at http://localhost:8000.

API Endpoint
The API exposes a single endpoint for making predictions.

POST /predict
This endpoint accepts a JSON payload with a sentence and returns the predicted intent and slots.

Request Body:

JSON

{
  "sentence": "book a flight to boston"
}
Sample Response:

The response includes the predicted intent and a list of slots corresponding to each token in the input sentence.

JSON

{
  "intent": "atis_flight",
  "slots": [
    "O",
    "O",
    "O",
    "O",
    "B-to_loc",
    "I-to_loc"
  ]
}
Dataset Information
This project is configured to work with the ATIS (Airline Travel Information System) and SNIPS datasets. The pre-processing step converts the raw data into a JSONL format, where each line is a JSON object with three keys:

tokens: A list of words in the sentence.

slots: A list of corresponding slot labels in IOB format.

intent: The intent label for the sentence.

Example JSONL line:

JSON

{"tokens": ["book", "a", "flight", "to", "boston"], "slots": ["O", "O", "O", "O", "B-to_loc"], "intent": "atis_flight"}

