# English–French Neural Machine Translation with Transformers

## Overview
This project is a **research-style demonstration** of a custom Transformer-based neural machine translation (NMT) model implemented from scratch in **PyTorch**.  
It builds an **end-to-end English-to-French translation system** that includes:

- Dataset preparation from the **OPUS-100 corpus**
- Word-level tokenizer training (with special tokens)
- Custom Transformer encoder–decoder architecture
- Advanced training features:
  - Label smoothing
  - Noam learning rate schedule with warmup
  - Mixed precision training (AMP)
  - Gradient clipping
  - Early stopping
- Evaluation with BLEU and chrF scores
- Beam search and greedy decoding for inference
- Checkpointing for every epoch and best model selection

The goal is to provide a **complete, reproducible, and educational** NMT pipeline that is useful both for understanding Transformer internals and as a portfolio example.

---

## Dataset
- **Name**: [OPUS-100 corpus](https://opus.nlpl.eu/opus-100.php)  
- **Languages**: English → French  
- **Content**: 100 languages translated to/from English; here we use the full English–French subset  
- **Size**: Entire available dataset used (train/dev/test splits)  
- **Format**: Plain-text `.en` / `.fr` sentence pairs converted to CSV

### Preparing the CSV files
Raw OPUS-100 text files in `data/en-fr/` are converted to CSV with:
```bash
python src/sourceData2CSV.py
```
This creates:
```bash
data/opus100_en_fr_train.csv
data/opus100_en_fr_val.csv
data/opus100_en_fr_test.csv
```
--- 

## Modle
### A custom implementation of the Transformer architecture with:
- Embedding dimension (d_model): 512
- Feed-forward dimension (d_ff): 2048
- Heads: 8
- Layers: 6 encoder + 6 decoder
- Dropout: 0.1
- Max sequence length: 512 tokens
- Beam size (inference): 4 by default
### Key Features
- Positional Encoding: Sinusoidal, auto-extended if needed
- Multi-Head Attention: Custom implementation
- Feed-Forward Network: Two-layer ReLU + dropout
- Masking:
  - Key padding mask
  - Subsequent mask for decoder
- Loss: CrossEntropy with label smoothing (0.1)

--- 
## Installation
### 1-  Clone the repository
```bash
git clone https://github.com/your-username/translation_EN-FR.git
cd translation_EN-FR
```

### 2- Create and activate a virtual environment
``` bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows
```

### 3- Install dependencies
``` bash
pip install -r requirements.txt
```

#### Tested with:|
- Python ≥ 3.10
- PyTorch 2.2.0
- CUDA 12.1+ (GPU recommended)

---

## Training
Run: 
```bash
python src/train.py
```

What it does:
- Loads CSV data
- Trains English & French tokenizers (WordLevel, vocab size = 32000)
- Builds Transformer model
- Trains for up to 30 epochs (early stopping if no val loss improvement for 5 epochs)
- Saves:
    - Per-epoch checkpoints in `models/`
    - Best checkpoint in `checkpoints/best.pt`
    - Final model in `models/transformer_attention_final.pt`

--- 

## Evlaluation
```bash
python src/eval.py
````
Computes:
- Validation loss
- Perplexity (PPL)
- Token accuracy
- BLEU score
- chrF2 score (if `sacrebleu` is installed)

--- 
 
## inference 
``` bash
python src/infer.py --src "This is a test sentence."

```
### Optional flags:
- ckpt: Path to checkpoint (.pt file or directory)
- beam: Beam size for decoding (default from config)
- max_len: Maximum output sequence length

Example:
``` bahs
python src/infer.py --src "Machine translation is fascinating." --beam 4
```

## Project structure
```php
translation_EN-FR/
├── checkpoints/              # Best model checkpoint
├── data/                      # Dataset and processed CSVs
│   ├── en-fr/                 # Raw OPUS100 files
│   ├── opus100_en_fr_train.csv
│   ├── opus100_en_fr_val.csv
│   ├── opus100_en_fr_test.csv
├── models/                    # Saved model checkpoints per epoch + final
├── src/                       # Source code
│   ├── config.py              # Hyperparameters & paths
│   ├── dataloader.py          # Tokenizer training & dataset class
│   ├── sourceData2CSV.py      # Converts raw data to CSV
│   ├── transformerModel.py    # Transformer architecture
│   ├── train.py               # Training loop
│   ├── eval.py                # Evaluation loop
│   ├── infer.py               # Inference (beam/greedy)
├── tokenizerfiles/            # Saved tokenizers
├── requirements.txt           # Dependencies
└── README.md                  # This file
```
### Repository Size Notice
The `data/` and `models/` directories are not synced to GitHub due to large file sizes and repository space limitations.
To reproduce results:
- Download the OPUS-100 English–French dataset manually from OPUS-100 and place it in `data/en-fr/`
- Models will be created automatically during training and saved in `models/` and `c`heckpoints/`
---

## Results

<<<(To be updated once training completes — include BLEU, chrF2, and token accuracy)>>>

``` ymal
- Val Loss: ??
- Val PPL: ??
- Val Token Accuracy: ??
- Val BLEU: ??
- Val chrF2: ??
```

---

## Achnowlegdement
- OPUS-100 dataset 
- PyTorch
- Parts of the code and design inspired by discussions with ChatGPT.





