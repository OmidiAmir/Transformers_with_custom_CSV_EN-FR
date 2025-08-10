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
        self.checkpoint_dir = os.path.join(self.PROJECT_ROOT, 'checkpoints')

        # Dataset
        self.dataSetFileName = 'opus100_en_fr.csv'
        self.sourceLang = 'en'
        self.targetLang = 'fr'

        # Model settings
        self.d_model = 256
        self.num_heads = 4
        self.num_layers = 2
        self.d_ff = 512
        self.dropout = 0.1
        self.max_seq_length = 128

        # Training settings
        self.trainBatchSize = 32
        self.valBatchSize = 1
        self.lr = 0.0003
        self.num_epochs = 30

        for p in [self.DATA_DIR, self.MODELS_DIR, self.TOK_DIR, self.checkpoint_dir]:
            os.makedirs(p, exist_ok=True)


config = Config()
