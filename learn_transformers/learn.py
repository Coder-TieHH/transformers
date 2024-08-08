from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd



data = pd.read_csv("/root/CODE/Datasets/ChnSentiCorp_htl_all.csv")
print(data.head())
