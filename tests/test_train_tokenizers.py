import os
from tokenizers import Tokenizer

def test_tokenizer_files_exist():
    print(os.getcwd())
    assert os.path.exists("tokenizers/tokenizer_en.json"), "English tokenizer not found"
    assert os.path.exists("tokenizers/tokenizer_fr.json"), "French tokenizer not found"

def test_tokenizer_can_tokenize_and_decode():
    tokenizer = Tokenizer.from_file("tokenizers/tokenizer_en.json")
    
    sample_text = "This is a test sentence."
    encoded = tokenizer.encode(sample_text)
    decoded = tokenizer.decode(encoded.ids)

    assert isinstance(encoded.ids, list), "Encoding failed"
    assert isinstance(decoded, str), "Decoding failed"
    assert len(encoded.ids) > 0, "Tokenizer returned empty encoding"
