import os
from pathlib import Path
import pandas as pd
import torch
import torchmetrics

from sklearn.model_selection import train_test_split
import pyarrow as pa
from datasets import Dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from torch.utils.data import DataLoader, random_split
from torch.utils.data import Dataset as TDataset

'''
Most of the code used in this file are taken from: 
https://github.com/hkproj/pytorch-transformer
'''


class AttrDict(dict):
  def __init__(self, *args, **kwargs):
    super(AttrDict, self).__init__(*args, **kwargs)
    self.__dict__ = self

def get_all_sentences(ds, lang):
    for item in ds:
        yield item[lang]

def get_or_build_tokenizer(ds, lang):
    tokenizer_path = Path("tokenizer_{0}.json".format(lang))
    if not Path.exists(tokenizer_path):
        # Most code taken from: https://huggingface.co/docs/tokenizers/quicktour
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

def validate_model(model, validation_ds, tokenizer_src, tokenizer_tgt, config):
    model.eval()
    count = 0
    source_texts = []
    expected = []
    predicted = []

    with torch.no_grad():
        for batch in validation_ds:
            count = count + 1
            encoder_input = batch["encoder_input"].to(config.device)
            encoder_mask = batch["encoder_mask"].to(config.device)

            # SOS, EOS, and PAD tokens
            sos_idx = torch.tensor([tokenizer_tgt.token_to_id("[SOS]")], dtype=torch.int64).to(config.device)
            eos_idx = torch.tensor([tokenizer_tgt.token_to_id("[EOS]")], dtype=torch.int64).to(config.device)
            pad_idx = torch.tensor([tokenizer_tgt.token_to_id("[PAD]")], dtype=torch.int64).to(config.device)

            # Initialize the decoder input with SOS and PAD tokens
            decoder_input = torch.cat(
                [
                    sos_idx,
                    torch.tensor([pad_idx] * (config.max_seq_length - 1), dtype=torch.int64).to(config.device),
                ],
                dim=0,
            ).unsqueeze(0).to(config.device)

            for i in range(config.max_seq_length - 1):
                decoder_mask = causal_mask(decoder_input.size(1)).type_as(encoder_mask).to(config.device)
                output, _, _ = model(encoder_input, decoder_input, encoder_mask, decoder_mask)
                prob = output.contiguous().view(-1, tokenizer_tgt.get_vocab_size())
                next_word = torch.argmax(prob[i + 1, :])
                decoder_input[0, i + 1] = next_word

                if next_word == eos_idx:
                    break

            model_out = decoder_input.squeeze(0)
            source_text = batch["src_text"][0]
            target_text = batch["tgt_text"][0]
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())

            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_out_text)

            # Print the source, target, and model output
            print(f"SOURCE: {source_text}")
            print(f"TARGET: {target_text}")
            print(f"PREDICTED: {model_out_text}")

            if count == 2:                
                break
    
    
    # Evaluate the character error rate
    # Compute the char error rate 
    metric = torchmetrics.CharErrorRate()
    cer = metric(predicted, expected)


    # Compute the word error rate
    metric = torchmetrics.WordErrorRate()
    wer = metric(predicted, expected)


    # Compute the BLEU metric
    metric = torchmetrics.BLEUScore()
    bleu = metric(predicted, expected)

    print('\n\n\t\t-----------------------------')
    print('\t\tCharacter Error Rate is: ', cer)
    print('\t\tWord Error Rate is:      ', wer)
    print('\t\tBlue score is:           ', bleu)