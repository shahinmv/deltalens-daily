import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd

nltk.download('vader_lexicon', quiet=True)
sia = SentimentIntensityAnalyzer()

def add_vader_sentiment(df_news):
    def score_row(row):
        text = f"{row['title']} . {row['description']}"
        scores = sia.polarity_scores(text)
        return scores['compound']
    df_news = df_news.copy()
    df_news['vader_score'] = df_news.apply(score_row, axis=1)
    return df_news

def aggregate_daily_sentiment(df_news):
    df_newsdaily_sentiment = (
        df_news
          .groupby(df_news['date'].dt.date)
          .agg({
              'vader_score':'mean',
              'title':'count'
          })
          .rename(columns={
              'vader_score':'avg_vader_compound',
              'title':'article_count'
          })
          .reset_index()
    )
    df_newsdaily_sentiment['date'] = pd.to_datetime(df_newsdaily_sentiment['date'])
    df_newsdaily_sentiment.set_index('date', inplace=True)
    df_newsdaily_sentiment.index = df_newsdaily_sentiment.index.date
    return df_newsdaily_sentiment 