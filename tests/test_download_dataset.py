import os
import pandas as pd

def test_dataset_download():
    file_path = os.path.join("data", "opus_books_en_fr.csv")
    assert os.path.exists(file_path), f"CSV file not found: {file_path}"

    df = pd.read_csv(file_path)
    assert "en" in df.columns and "fr" in df.columns, "Missing expected columns"
    assert len(df) == 10000, f"Dataset size is incorrect: {len(df)}"
