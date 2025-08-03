import torch
import torch.nn as nn
import torch.utils.data as data
import os
from tqdm import tqdm
import mlflow
import mlflow.pytorch

from src.config import config
from src.models.transformer_model import Transformer
from src.data.dataloader import get_data_ready
from src.utils.helper import validate_model

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

# Data preparation
train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_data_ready(config)

model = Transformer(tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size(), config)
model.to(config.device)

criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, betas=(0.9, 0.98), eps=1e-9)

# MLflow: start logging
with mlflow.start_run():

    # Log config parameters
    mlflow.log_params({
        "epochs": config.num_epochs,
        "learning_rate": config.lr,
        "batch_size": config.trainBatchSize,
        "max_seq_length": config.max_seq_length,
        "source_language": config.sourceLang,
        "target_language": config.targetLang
    })

    for epoch in range(config.num_epochs):
        model.train()
        total_loss = 0.0

        batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch + 1:02d}")
        for batch in batch_iterator:
            optimizer.zero_grad()

            encoder_input = batch['encoder_input'].to(config.device)
            decoder_input = batch['decoder_input'].to(config.device)
            encoder_mask = batch['encoder_mask'].to(config.device)
            decoder_mask = batch['decoder_mask'].to(config.device)
            label = batch['label'].to(config.device)

            output, _, _ = model(encoder_input, decoder_input, encoder_mask, decoder_mask)

            loss = criterion(output.contiguous().view(-1, tokenizer_tgt.get_vocab_size()), label.contiguous().view(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        average_loss = total_loss / len(train_dataloader)
        print(f"Epoch: {epoch + 1}, Loss: {average_loss}")
        mlflow.log_metric("train_loss", average_loss, step=epoch)

        # Save the model at the end of every epoch
        model_dir = f'{config.currentPath}/model'
        os.makedirs(model_dir, exist_ok=True)
        model_path = f'{model_dir}/Transformer_EN_FR_epoch{epoch + 1}.pt'
        torch.save(model.state_dict(), model_path)
        mlflow.log_artifact(model_path)

        # Validation and log metrics
        cer, wer, bleu = validate_model(model, val_dataloader, tokenizer_src, tokenizer_tgt, config, log=False)
        mlflow.log_metrics({
            "cer": cer,
            "wer": wer,
            "bleu": bleu
        }, step=epoch)

print("Training finished.")
