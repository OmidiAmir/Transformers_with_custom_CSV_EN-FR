import os
import torch

class Config:
    def __init__(self):
        # System
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.seed = 42                                      # ADDED
        self.mixed_precision = True                         # ADDED (AMP on/off)

        # Paths
        self.PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self.DATA_DIR = os.path.join(self.PROJECT_ROOT, "data")
        self.MODELS_DIR = os.path.join(self.PROJECT_ROOT, "models")
        self.TOK_DIR = os.path.join(self.PROJECT_ROOT, "tokenizerfiles")
        self.checkpoint_dir = os.path.join(self.PROJECT_ROOT, "checkpoints")

        # Dataset
        self.dataSetFileName = "opus100_en_fr.csv"
        self.sourceLang = "en"
        self.targetLang = "fr"

        # Model (CHANGED to a solid Transformer-Base)
        self.d_model = 512          # CHANGED (was 256->512)
        self.num_heads = 8          # CHANGED (was 4->8)
        self.num_layers =6         # CHANGED (was 2->6)
        self.d_ff = 2048            # CHANGED (was 512->2048)
        self.dropout = 0.1
        self.max_seq_length = 512

        # Training
        self.trainBatchSize = 32
        self.valBatchSize = 32       # CHANGED (was 1)
        self.lr = 5e-4               # CHANGED (was 3e-4) — pairs well with warmup/plateau
        self.num_epochs = 30         # CHANGED (was 100) — use early stopping instead
        self.grad_clip = 1.0         # ADDED
        self.label_smoothing = 0.1   # ADDED (use in CrossEntropyLoss)
        self.patience = 5            # ADDED (early stopping on val loss)
        self.warmup_steps = 4000     # ADDED (if you add Noam scheduler later)

        # Inference
        self.beam_size = 4           # ADDED
        self.max_gen_len = 128       # ADDED

        for p in [self.DATA_DIR, self.MODELS_DIR, self.TOK_DIR, self.checkpoint_dir]:
            os.makedirs(p, exist_ok=True)

config = Config()
