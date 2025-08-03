from tokenizers import Tokenizer, models, trainers, pre_tokenizers
import pandas as pd
import os

def get_all_sentences(df, lang):
    return df[lang].dropna().astype(str).tolist()

def train_and_save_tokenizer(sentences, tokenizer_path, vocab_size=8000):
    tokenizer = Tokenizer(models.WordLevel(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.WordLevelTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=["[PAD]", "[UNK]", "[SOS]", "[EOS]"]
    )
    tokenizer.train_from_iterator(sentences, trainer=trainer)
    tokenizer.save(tokenizer_path)
    print(f"Tokenizer saved at: {tokenizer_path}")

if __name__ == "__main__":
    df = pd.read_csv("data/opus_books_en_fr.csv")

    os.makedirs("tokenizers", exist_ok=True)

    en_sentences = get_all_sentences(df, "en")
    fr_sentences = get_all_sentences(df, "fr")

    train_and_save_tokenizer(en_sentences, "tokenizers/tokenizer_en.json")
    train_and_save_tokenizer(fr_sentences, "tokenizers/tokenizer_fr.json")
