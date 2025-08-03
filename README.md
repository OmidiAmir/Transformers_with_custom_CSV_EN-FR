#   English–French Translation with Transformers + MLOps
This project builds an end-to-end English-to-French neural machine translation system using a custom Transformer architecture in PyTorch. It includes dataset preparation, tokenizer training, model training, and MLOps components like MLflow, Docker, and API deployment.


## 📥 Dataset
This project uses the `opus_books` English–French dataset from Hugging Face.

To download and save a 10k-sample CSV:
```bash
python ./src/data/download_dataset.py
```

Saved to: data/opus_books_en_fr.csv

---

## 🔠 Tokenization
The project uses a custom WordLevel tokenizer (trained from scratch) for both English and French.

To train and save the tokenizers:

```bash
python ./src/data/train_tokenizers.py
````
This will generate two files:
- tokenizers/tokenizer_en.json
- tokenizers/tokenizer_fr.json

To verify the tokenizers work, run:

```bash
pytest tests/test_train_tokenizers.py
```



## 🧠 Transformer Model
This project implements a custom Transformer model from scratch based on the "Attention Is All You Need" paper.  
It includes:

- Multi-head self-attention mechanism
- Sinusoidal positional encoding
- Encoder and decoder layers with LayerNorm, residual connections, and dropout
- Custom masking for source and target sequences

The model is defined in `src/models/transformer_model.py`.

To run the unit test:

```bash
pytest tests/test_transformer_model.py
``` 
The test checks:

- Model instantiation
- Forward pass with dummy data
- Output shapes match expectations

## 🏋️‍♂️ Training
_(Coming soon)_

## 📊 Experiment Tracking (MLflow)
_(Coming soon)_

## 🚀 Deployment (API)
_(Coming soon)_

## 📦 Installation
```bash
pip install -r requirements.txt
``` 
---
## 💻 How to Run
_(Coming soon)_

## 📈 Results
_(Coming soon)_
 
