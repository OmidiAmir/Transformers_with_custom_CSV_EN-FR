import os
import torch

class Config:
    def __init__(self):
        # System settings
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

        self.DATA_DIR = os.path.join(self.PROJECT_ROOT, "data")
        
        self.MODELS_DIR = os.path.join(self.PROJECT_ROOT, "models")

        self.TOK_DIR = os.path.join(self.PROJECT_ROOT, "tokenizerfiles")

        # Dataset
        self.dataSetFileName = 'opus100_en_fr.csv'
        self.sourceLang = 'en'
        self.targetLang = 'fr'
        self.trainPercentage = 0.9

        # Model settings
        self.d_model = 64
        self.num_heads = 4
        self.num_layers = 1
        self.d_ff = 256
        self.dropout = 0.0
        self.max_seq_length = 128

        # Training settings
        self.trainBatchSize = 16
        self.valBatchSize = 1
        self.lr = 0.0007
        self.num_epochs = 10

        # Checkpoint path (optional now)
        self.checkpoint_dir = os.path.join(self.PROJECT_ROOT, 'checkpoints')

config = Config()
