import pandas as pd
import os
import shutil

datasets = ['aircrowd6', 'fej2013', 'valence5', 'valence7', 'WS', 'bluebird', 'rte', 'ZenCrowd_all', 'ZenCrowd_in', 'ZenCrowd_us', 'CF', 'CF_amt', 'fact_eval', 'MS', 's4_Dog_data', 's4_Face_Sentiment_Identification',
              's5_AdultContent', 'web', 'd_jn-product', 'd_sentiment', 'SP', 'SP_amt', 'trec', 'sentiment']
for dataset in datasets:
    os.mkdir('datasets/' + dataset)
    clean_data = pd.read_csv('datasets_original/' + dataset + '/label.csv').drop_duplicates(
        subset=['item', 'worker'], keep='first')
    clean_data.to_csv('datasets/' + dataset + '/label.csv', index=False)
    shutil.copy('datasets_original/' + dataset + '/truth.csv', 'datasets/' + dataset + '/truth.csv')
    print(dataset + ' Done')

