from datasets import load_dataset
import pandas as pd
import os

def download_opus_books(output_csv_path="data/opus_books_en_fr.csv", num_samples=None):
    # Load the dataset
    dataset = load_dataset("opus_books", "en-fr")

    # Convert train split to pandas dataframe
    data = dataset["train"]
    data_list = []

    for item in data:
        if item["translation"]["en"] and item["translation"]["fr"]:
            data_list.append({
                "en": item["translation"]["en"],
                "fr": item["translation"]["fr"]
            })

    df = pd.DataFrame(data_list)

    if num_samples:
        df = df.sample(n=num_samples, random_state=42).reset_index(drop=True)

    # Create data folder if not exists
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)

    # Save to CSV
    import csv
    df.to_csv(output_csv_path, index=False, encoding='utf-8', quoting=csv.QUOTE_ALL)
    print(f"Saved {len(df)} sentence pairs to: {output_csv_path}")


if __name__ == "__main__":
    download_opus_books(num_samples=10000)  # You can change sample size if needed
