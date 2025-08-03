


import torch
import torch.nn as nn
import torch.utils.data as data
import os
# from pathlib import Path
from tqdm import tqdm

### comfiguration parameters
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

model.train()

for epoch in range(config.num_epochs):
        model.train()
        total_loss = 0.0

        batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch + 1:02d}")
        for batch in batch_iterator:
            optimizer.zero_grad()

            encoder_input = batch['encoder_input'].to(config.device)
            decoder_input = batch['decoder_input'].to(config.device)

            label = batch['label'].to(config.device)

            output, _, _ = model(encoder_input, decoder_input)

            loss = criterion(output.contiguous().view(-1, tokenizer_tgt.get_vocab_size()), label.contiguous().view(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        average_loss = total_loss / len(train_dataloader)
        print(f"Epoch: {epoch + 1}, Loss: {average_loss}")

        # Save the model at the end of every epoch
        os.makedirs(f'{config.currentPath}/model', exist_ok=True)
        torch.save(model.state_dict(), f'{config.currentPath}/model/TrainedTransformerModelTranslateing_{config.sourceLang}_to_{config.targetLang}_epoch{epoch + 1}.pt')

        # Run validation at the end of every epoch
        validate_model(model, val_dataloader, tokenizer_src, tokenizer_tgt, config)

print("Training finished.")