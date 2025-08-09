import pandas as pd
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")


en_path_train = os.path.join(DATA_DIR,"en-fr","opus.en-fr-train.en")
fr_path_train = os.path.join(DATA_DIR,"en-fr","opus.en-fr-train.fr")

en_path_test = os.path.join(DATA_DIR,"en-fr","opus.en-fr-test.en")
fr_path_test = os.path.join(DATA_DIR,"en-fr","opus.en-fr-test.fr")

en_path_val = os.path.join(DATA_DIR,"en-fr","opus.en-fr-dev.en")
fr_path_val = os.path.join(DATA_DIR,"en-fr","opus.en-fr-dev.fr")

with open(en_path_train, encoding="utf-8") as f_en, open(fr_path_train, encoding="utf-8") as f_fr:
    en_lines_train = [line.strip() for line in f_en]
    fr_lines_train = [line.strip() for line in f_fr]

with open(en_path_test, encoding="utf-8") as f_en, open(fr_path_test, encoding="utf-8") as f_fr:
    en_lines_test = [line.strip() for line in f_en]
    fr_lines_test = [line.strip() for line in f_fr]


with open(en_path_val, encoding="utf-8") as f_en, open(fr_path_val, encoding="utf-8") as f_fr:
    en_lines_val = [line.strip() for line in f_en]
    fr_lines_val = [line.strip() for line in f_fr]


assert len(en_lines_train) == len(fr_lines_train), "Mismatched line counts in train dataset!"
assert len(en_lines_test) == len(fr_lines_test), "Mismatched line counts in test dataset!"
assert len(en_lines_val) == len(fr_lines_val), "Mismatched line counts in validation dataset!"

df_train = pd.DataFrame({'en': en_lines_train, 'fr': fr_lines_train})
df_test = pd.DataFrame({'en': en_lines_test, 'fr': fr_lines_test})
df_val = pd.DataFrame({'en': en_lines_val, 'fr': fr_lines_val})

df_train.to_csv(os.path.join(DATA_DIR,"opus100_en_fr_train.csv"), index=False)
print("Saved to data/en-fr/opus100_en_fr_train.csv")

df_test.to_csv(os.path.join(DATA_DIR,"opus100_en_fr_test.csv"), index=False)
print("Saved to data/en-fr/opus100_en_fr_test.csv")

df_val.to_csv(os.path.join(DATA_DIR,"opus100_en_fr_val.csv"), index=False)
print("Saved to data/en-fr/opus100_en_fr_val.csv")