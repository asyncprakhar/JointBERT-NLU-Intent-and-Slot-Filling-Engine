# JointBERT NLU: Intent and Slot Filling Engine
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
## Overview🚀

<p align="justify">
<strong>JointBERT NLU</strong> is a powerful, production-ready Natural Language Understanding (NLU) engine designed to be the core "brain" of any conversational AI. This project exists to solve a common challenge in building chatbots and voice assistants: creating an NLU component that is both highly accurate and efficient.
</p>

Instead of treating intent recognition and slot filling as separate problems, this engine uses a sophisticated joint-learning approach. It leverages a pre-trained **BERT** model as a shared encoder for two downstream tasks:

An **intent classification** head to identify the user's overall goal.

A **slot filling head** to extract key information (like dates, names, or locations) on a token-by-token basis.

By training these two heads simultaneously, the model learns to perform both tasks in a single, efficient pass, significantly improving performance and simplifying the entire NLU pipeline.

This project is for **developers**, **data scientists**, and **researchers** who need to build, deploy, or simply understand the mechanics of a modern NLU system. Its importance lies in providing a complete, end-to-end toolkit—from training scripts to a production-ready API—that makes state-of-the-art natural language understanding accessible and practical for any application. 🤖


# The Core (Getting Started)

## ✨ Key Features
- **State-of-the-Art Joint Model**: Utilizes a **JointBERT** architecture with a shared BERT encoder and two distinct heads for intent and slots. This makes predictions highly efficient and more accurate than separate models.

- **Complete End-to-End Workflow**: Comes with a full suite of scripts to **train** the model from scratch, **evaluate** its performance, and make quick **predictions** directly from the command line.

- **Production-Ready API**: Includes a **high-performance REST API** built with FastAPI and Uvicorn, ready to be deployed as a standalone microservice for real-time inference.

- **Train on Custom Datasets**: The training pipeline is built to ingest any dataset, provided it's formatted into the required **JSONL structure**. This allows you to easily train the model for your specific domain. The underlying architecture also supports different Transformer backbones (like RoBERTa or ALBERT) with minor adjustments.

- **Comprehensive Evaluation**: The evaluation script reports on key NLU metrics, including **Intent Accuracy**, token-level **Slot F1-score**, and overall **Joint Accuracy** for a complete picture of model performance.

- **Self-Contained & Reproducible**: After training, the project automatically saves all necessary components—the model weights, tokenizer configuration, and label mappings—into a single `artifacts/` directory.  This ensures that your model's performance is perfectly reproducible and makes it incredibly simple to deploy for inference on another machine or in a container.

## 🛠️ Tech Stack
This project leverages a modern Python stack for building, training, and serving deep learning models. The core technologies are:

- Python: The primary language used for all the code.

- PyTorch: The main deep learning framework for building and training the JointBERT model.

- Hugging Face Transformers: Provides the pre-trained BERT models and state-of-the-art tokenizers that are the foundation of the NLU engine.

- FastAPI: A high-performance web framework used to create the efficient and easy-to-use prediction API.

- Uvicorn: The lightning-fast ASGI server responsible for running the FastAPI application in production.

- Seqeval: A specialized library for evaluating the model's performance on the slot-filling task using standard metrics like F1-score.


## 🚀Installation
This step-by-step guide will get you from a fresh clone to a ready-to-run application.

1. Clone the Repository
    
    Open your terminal and clone the project repository using Git.
    ```
    git clone https://github.com/asyncprakhar/JointBERT-NLU-Intent-and-Slot-Filling-Engine.git
    ```
2. Navigate to the Project Directory
    ```
    cd JointBERT-NLU
    ```
3. Create and Activate a Virtual Environment

    It's a best practice to create a virtual environment to manage project-specific dependencies.
    ```
    # Create the environment
    python -m venv venv

    # Activate the environment
    # On macOS/Linux:
    source venv/bin/activate

    # On Windows:
    # .\venv\Scripts\activate
    ```
    You'll know it's active when you see (venv) at the beginning of your terminal prompt.

4. Install Dependencies
    
    Install all the required Python packages from the requirements.txt file.
    ```
    pip install -r requirements.txt
    ```
    You are now all set to run the project! 🎉

## 🏃‍♀️ Usage
The project provides multiple modes of operation: training a new model, evaluating a trained one, making a quick prediction from the terminal, or running a live API server.
***
### ⚠️ Important: Data Configuration
To use your own data, the project requires **three separate files** (training, validation, and evaluation) in a specific **JSONL format**. Each line must be a JSON object with `"tokens",` `"slots",` and `"intent"` keys.

You have two options for configuring the data paths:
***
#### Option 1: The Easy Way (Recommended)

Simply name your files as specified below and place them directly inside the `data/` folder. **No code changes are needed** if you follow this convention.

- `train_data.jsonl` (for your training dataset)

- `val_data.jsonl` (for your validation dataset)

- `test_data.jsonl` (for your evaluation dataset)
***
#### Option 2: The Flexible Way
If you prefer to keep your data in a different location or have a custom naming scheme, you can edit the path inside the `src/scripts/main.py` file.

- Training & Validation Paths: Modify the paths inside the `load_tokenizer_and_data` function. [Go to line](src/scripts/main.py#L90-L91)

- Evaluation Path: Modify the path inside the `run_eval` function. [Go to line](src/scripts/main.py#L173)

***Again Strict JSONL Format***: *All data files must be in the strict JSONL format. Each line must be a JSON object containing the keys: "tokens", "slots", and "intent".*
***
### 📖 Data Format Details
All data files (train_data.jsonl, val_data.jsonl, etc.) must follow a strict format to be compatible with the tokenizer and training scripts. Each line in your data files must be a single JSON object with three required keys:

```
{
  "tokens": ["show", "me", "flights", "to", "new", "york"],
  "slots": ["O", "O", "O", "O", "B-location", "I-location"],
  "intent": "find_flight"
}
```

#### Key Breakdown
- `"tokens"`: A list of strings, where each string is a single word (token) in the input sentence.

- `"intent"`: A single string representing the overall purpose of the sentence.

- `"slots"`: A list of label strings. This list must have the exact same length as the "tokens" list. It uses the standard IOB (Inside, Outside, Beginning) notation for entity tagging:    
    - `O`: The token is Outside of any entity.

    - `B-<entity_type>`: The token marks the Beginning of an entity (e.g., B-location for "new").

    - `I-<entity_type>`: The token is Inside an entity and follows a B- tag. This is used for multi-word entities (e.g., I-location for "york").

___
### 1. Training the Model
This command trains the `JointBERT` model on a specified dataset. After completion, it saves the trained model weights, tokenizer configuration, and label mappings into the `artifacts/` directory.
```
python src/scripts/main.py train
```
*Note: The dataset path and training hyperparameters like epochs or learning rate are currently set within the `src/scripts/main.py` script. You can modify them there to customize your training run.*
___

### 2. Evaluating the Model
This command loads the saved model from the `artifacts/` directory and evaluates its performance on the test dataset. It will print a report with key metrics.
```
python src/scripts/main.py eval
```
You should see an output detailing the **Intent Accuracy**, **Slot F1-Score**, and **Joint Accuracy**.
___
### 3. Predicting with the CLI
Use this mode for a quick test on a single sentence. It loads the trained model and prints the predicted intent and slots directly to your console.

Pass the sentence you want to test in quotes.

```
python src/scripts/main.py predict "book a flight from london to new york"
```
___

### 4. Running the API Server
This command starts the high-performance web server, making your NLU engine available through a REST API.
```
uvicorn src.api.inference:app --host 0.0.0.0 --port 8000 --reload
```
Once you run this command, the API will be live and accessible. You can now send POST requests to `http://127.0.0.1:8000/predict` to get real-time predictions. See the next section for details on the API endpoint.

# The Details(Diving Deeper)

## ⚙️ Configuration
This project is configured by modifying constants directly within the Python scripts, not through environment variables. The most important settings are located at the top of the relevant scripts for easy access.

### Training & Evaluation (`src/scripts/main.py`)

All primary training and evaluation settings are defined as constants at the top of the src/scripts/main.py file. You can customize the model and training loop by changing these values:

- `TOKENIZER_NAME`: The Hugging Face identifier for the pre-trained model you want to use (e.g., "bert-base-uncased", "roberta-base").

- `MAX_LEN`: The maximum sequence length for sentences. Longer sentences are truncated, and shorter ones are padded.

- `BATCH_SIZE`: The number of examples processed in a single forward/backward pass. Adjust this based on your GPU memory.

- `EPOCHS`: The total number of times the training loop will iterate over the entire training dataset.

- `LEARNING_RATE`: The step size for the AdamW optimizer, which controls how much the model's weights are updated during training.

***

### API Server (`src/api/inference.py`)
The API server configuration is handled at the top of `src/api/inference.py`. The main parameter you might change is the path to the model artifacts if you store them in a non-default location:

`ARTIFACTS_DIR`: The root directory where the API server looks for the trained `model`, `tokenizer`, and label mappings.

## Project Structure

```
JointBERT-NLU/
│
├── data/
│
├── artifacts/
│   ├── saved_weights/
│   │   └── saved_model.pt
│   │
│   ├── tokenizer/
│   │   ├── special_tokens_map.json
│   │   ├── tokenizer_config.json 
│   │   ├── tokenizer.json 
│   │   └── vocab.txt               
│   │
│   ├── intent2id.json           
│   └── slot2id.json           
│
├── src/                       
│   ├── models/
│   │   ├── __init__.py
│   │   ├── loss.py
│   │   └── model_module.py
│   │
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── metrics.py
│   │   ├── model_io.py
│   │   └── tokenizer_module.py
│   │
│   ├── data_processors/
│   │   └── dataset_module.py
│   │
│   └── scripts/
│       ├── train.py
│       ├── predict.py
│       ├── evaluate.py
│       └── main.py
│
├── README.md
└── requirements.txt
```

### Directory Breakdown
- artifacts/: Contains all outputs from the training process. This includes the saved model weights, the trained tokenizer, and the label-to-ID mappings needed for inference.

- data/: Holds all datasets, including the raw data (e.g., ATIS, SNIPS) and any preprocessed versions.

- src/: The main container for all Python source code.

- api/: The FastAPI application for serving the trained model via a REST API.

- data_processors/: Contains the PyTorch Dataset wrapper that prepares data for the model.

- models/: Defines the JointBERT model architecture and the custom loss function.

- scripts/: Includes the command-line scripts. main.py is the entry point for training, evaluation, and prediction.

- utils/: A collection of helper modules for tasks like tokenization, metric calculations, and file I/O.

- requirements.txt: A list of all Python dependencies required to run the project.

## 🔌 API Reference
The API provides a single endpoint to get real-time intent and slot predictions from a sentence.
**POST** `/predict`
This endpoint analyzes the provided sentence and returns its predicted intent and a list of tokens with their corresponding slot labels.

**Request Body:**

The request body should be a JSON object with a single key, `"sentence"`.
```
{
  "sentence": "book a flight from london to new york"
}
```

**Success Response (200 OK):**

The response will be a JSON object containing the predicted intent and a list of slots. Each item in the slots list is a pair containing the token and its predicted IOB slot tag.

```
{
  "intent": "atis_flight",
  "slots": [
    ["book", "O"],
    ["a", "O"],
    ["flight", "O"],
    ["from", "O"],
    ["london", "B-fromloc.city_name"],
    ["to", "O"],
    ["new", "B-toloc.city_name"],
    ["york", "I-toloc.city_name"]
  ]
}
```

# The Community (Collaboration)

## 🤝 Contributing

Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📜 License
This project is distributed under the MIT License. See the [`LICENSE`](./LICENSE) file for more information.

## 📧 Contact
Prakhar Srivastava - [LinkedIn](www.linkedin.com/in/asyncioprakhar) | [Email Me](srivastavaprakhar081@gmail.com)

Project Link: https://github.com/asyncprakhar/JointBERT-NLU-Intent-and-Slot-Filling-Engine