import os
import torch

class Config:
    def __init__(self):
        # System settings
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.currentPath = os.path.dirname(os.path.abspath(__file__))

        # Dataset
        self.dataSetFileName = 'english_french_sentences.csv'
        self.sourceLang = 'en'
        self.targetLang = 'fr'
        self.trainPercentage = 0.9

        # Model settings
        self.d_model = 512
        self.num_heads = 8
        self.num_layers = 6
        self.d_ff = 2048
        self.dropout = 0.1
        self.max_seq_length = 100

        # Training settings
        self.trainBatchSize = 8
        self.valBatchSize = 1
        self.lr = 0.0001
        self.num_epochs = 20

        # Checkpoint path (optional now)
        self.checkpoint_dir = os.path.join(self.currentPath, '..', 'checkpoints')

config = Config()
