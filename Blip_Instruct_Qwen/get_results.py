import pandas as pd
df=pd.read_csv('vqa_eval_results.csv')
blue=df['bleu_score'].mean()
bert=df['bert_score'].mean()
print("Bleu: ",blue)
print("Bert:",bert)