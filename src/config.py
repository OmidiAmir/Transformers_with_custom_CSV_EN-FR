import os
import torch

class Config:
    def __init__(self):
        # System
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.seed = 42                                      
        self.mixed_precision = True                         

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

        self.d_model = 512          
        self.num_heads = 8          
        self.num_layers =6         
        self.d_ff = 2048           
        self.dropout = 0.1
        self.max_seq_length = 512

        # Training
        self.trainBatchSize = 32
        self.valBatchSize = 32       
        self.lr = 5e-4               
        self.num_epochs = 30         
        self.grad_clip = 1.0         
        self.label_smoothing = 0.1   
        self.patience = 5            
        self.warmup_steps = 4000     

        # Inference
        self.beam_size = 4           
        self.max_gen_len = 128       

        for p in [self.DATA_DIR, self.MODELS_DIR, self.TOK_DIR, self.checkpoint_dir]:
            os.makedirs(p, exist_ok=True)

config = Config()
