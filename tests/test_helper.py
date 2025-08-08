import pytest
from src.config import config
from src.utils.helper import get_or_build_tokenizer
import pandas as pd
import pyarrow as pa
from datasets import Dataset

def test_tokenizer_build_and_load():
    dummy_data = pd.DataFrame({
        config.sourceLang: ["Hello world"],
        config.targetLang: ["Bonjour le monde"]
    })
    ds = Dataset(pa.Table.from_pandas(dummy_data))
    
    tokenizer_src = get_or_build_tokenizer(ds, config.sourceLang)
    tokenizer_tgt = get_or_build_tokenizer(ds, config.targetLang)
    
    assert tokenizer_src.get_vocab_size() > 0
    assert tokenizer_tgt.get_vocab_size() > 0
