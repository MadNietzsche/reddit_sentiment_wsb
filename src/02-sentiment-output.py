import pandas as pd
from transformers import AutoModelForSequenceClassification, pipeline
from typing import AnyStr

df_posts = pd.read_csv('../data/wallstreetbets.zip')
df_comments = pd.read_csv('../data/all_comments.zip')
sentiment_model = pipeline(model='mwkby/distilbert-base-uncased-sentiment-reddit-crypto')


class SentimentOutput(object):
    def __init__(self, df, model, kind, col, n_col):
        self.df = df
        self.model = model
        self.kind = kind
        self.col = col
        self.n_col = n_col

        self.text_parser()
        self.sentiment_cols()

    def text_parser(self) -> None:
        self.df[self.n_col] = self.df[self.col].fillna('')
        self.df[self.n_col] = self.df[self.n_col].str.replace(r'[\([{})\]]', '', regex=True)
        self.df[self.n_col] = self.df[self.n_col].str.replace(r'https?:\/\/.*[\r\n]*', '', regex=True)
        self.df[self.n_col] = self.df[self.n_col].str.replace(r'\n', '', regex=True)
        self.df[self.n_col] = self.df[self.n_col].str.replace(r'\\\*', '', regex=True)

        if self.kind == 'posts':
            self.df[self.n_col] = self.df['title'].str.cat(self.df[self.n_col], sep=' ')

    def sentiment_cols(self) -> None:
        sent_ls = self.model(self.df[self.n_col].tolist())
        self.df['sent_label'] = [dct['label'] for dct in sent_ls]
        self.df['sent_score'] = [dct['score'] for dct in sent_ls]
        self.df = self.df.drop(self.n_col, axis=1)


df_posts = SentimentOutput(df_posts, model=sentiment_model, kind='posts', col='body', n_col='n_body')

sent_comment = sentiment_model(df_comments['comments'].tolist())
