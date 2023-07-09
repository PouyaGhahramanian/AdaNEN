
"""
    Run this module by specifying dataset name
    to preprocess data and create data stream.
    -----------------------
    Pouya Ghahramanian
"""

import numpy as np
import pandas as pd
import logging
import sister
import argparse

logger = logging.getLogger()
logger.setLevel(logging.INFO)

DATA_PATH = 'Data/'
DATA_FOLDER = ''

parser = argparse.ArgumentParser(description='Get dataset and output file info.')
parser.add_argument('--dataset', '-d', default = 'nyt', help = 'Name of the dataset to be preprocessed.')
parser.add_argument('--output', '-o', default = 'nyt_modified_base', help = 'Name of the output file to be generated.')
args = parser.parse_args()
dataset = args.dataset
dataset_modified = args.output

df_out = pd.DataFrame()

embedder_w2v = sister.MeanEmbedding(lang = 'en')
embedder_bert = sister.BertEmbedding(lang = 'en')

# NYT
if dataset == 'nyt':
    df_out = pd.DataFrame([['Dummy Text!', np.zeros((300)), np.zeros((768)), 0.]], columns=['Text', 'W2V', 'Bert', 'Label'])
    DATA_FOLDER = 'nyt/'
    logging.info('NewYorkTimes dataset is selected...')
    df_train = pd.read_csv(DATA_PATH + DATA_FOLDER + 'nyt_train.csv')
    df_test = pd.read_csv(DATA_PATH + DATA_FOLDER + 'nyt_test.csv')
    df_all = df_train.append(df_test)
    data_size = df_all.values.shape[0]
    df_all = df_all.sort_values(by = 'pub_date')
    df_all['text'] = df_all['headline'] + ' ' + df_all['abstract']
    df_all.dropna(subset = ['text'], inplace = True)
    articles = list(df_all['text'].values)
    for i in range(data_size):
        if i%100 == 0:
            logging.info('Row #: ' + str(i))
        try:
            fullText = articles[i]
            embedding_w2v = embedder_w2v(articles[i])
            embedding_bert = embedder_bert(articles[i])
            label = df_all['is_popular'].values[i]
            newRow = {'Text': fullText, 'W2V': embedding_w2v, 'Bert': embedding_bert, 'Label': label}
            df_out = df_out.append(newRow, ignore_index = True)
        except Exception as e:
            logger.info('Exception at data item # {}'.format(str(i)))
            pass
    print(df_out.shape)
    print(df_out.head(10))
    print(df_out.columns)
    df_out.drop([0], axis = 0, inplace = True)
    logger.info('Preprocessing Completed. Saving Modified Dataset...')
    df_out.to_pickle(DATA_PATH + dataset_modified + '.csv')
    logger.info('Done!')

else:
    print('Selected dataset is not in the list. Try again...')
