import os
from wordcloud import WordCloud

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STOPWORDS_PATH = os.path.join(BASE_DIR, "stop_hinglish.txt")

from urlextract import URLExtract
from wordcloud import WordCloud
import pandas as pd
from collections import Counter
import emoji
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
from sklearn.cluster import DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

extractor = URLExtract()

MODEL = "j-hartmann/emotion-english-distilroberta-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

# Keep sentiment analysis if needed
def analyzesenti(text):
    encoded_text = tokenizer(text[:512], return_tensors='pt')
    output = model(**encoded_text)
    scores = output[0][0].detach().numpy()
    prob = softmax(scores)

    labels = model.config.id2label
    sentiment = labels[prob.argmax()]
    confidence = prob[prob.argmax()]

    return sentiment, confidence

def fetchstats(selected_user, df):
    if selected_user != "Overall":
        df = df[df['user'] == selected_user]

    num_msg = df.shape[0]
    words = [word for message in df['message'] for word in message.split()]

    num_img_msg = df[df['message'].str.contains('image omitted', case=False, na=False)].shape[0]
    num_aud_msg = df[df['message'].str.contains('audio omitted', case=False, na=False)].shape[0]

    links = [link for message in df['message'] for link in extractor.find_urls(message)]

    return num_msg, len(words), num_img_msg, num_aud_msg, len(links)

def busiest_users(df):
    busy = df['user'].value_counts().head()
    df_percent = ((df['user'].value_counts() / df.shape[0]) * 100).round(2).reset_index()
    df_percent.columns = ['name', 'percent']

    return busy, df_percent

import os
from wordcloud import WordCloud

# Path to stopwords file relative to helper.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STOPWORDS_PATH = os.path.join(BASE_DIR, "stop_hinglish.txt")

def create_wordcloud(selected_user, df):
    stop_words = set()
    try:
        with open(STOPWORDS_PATH, "r", encoding="utf-8") as f:
            stop_words = set(f.read().split())
    except FileNotFoundError:
        print("⚠️ stop_hinglish.txt not found. Proceeding with empty stopwords.")

    if selected_user != "Overall":
        df = df[df['user'] == selected_user]

    temp = df[df['message'] != 'image omitted']

    if temp.empty:
        return WordCloud(width=500, height=500, background_color="white").generate("No Data")

    temp['message'] = temp['message'].apply(
        lambda message: " ".join(
            word for word in message.lower().split() if word not in stop_words
        )
    )

    wc = WordCloud(width=500, height=500, min_font_size=10, background_color="white")
    df_wc = wc.generate(temp['message'].str.cat(sep=" "))
    return df_wc

def most_common_words(selected_user, df):
    stop_words = set()
    try:
        with open(STOPWORDS_PATH, "r", encoding="utf-8") as f:
            stop_words = set(f.read().split())
    except FileNotFoundError:
        print("⚠️ stop_hinglish.txt not found. Proceeding with empty stopwords.")

    if selected_user != "Overall":
        df = df[df['user'] == selected_user]

    temp = df[df['message'] != 'image omitted']

    words = [
        word for message in temp['message']
        for word in message.lower().split()
        if word not in stop_words
    ]

    if not words:
        return pd.DataFrame(columns=["Word", "Frequency"])

    return pd.DataFrame(Counter(words).most_common(20), columns=["Word", "Frequency"])

def emoji_helper(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    emojis = [c for message in df['message'] for c in message if emoji.is_emoji(c)]

    return pd.DataFrame(Counter(emojis).most_common(), columns=['Emoji', 'Count'])

def monthly_timeline(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    timeline = df.groupby(['year', 'month_num', 'month']).count()['message'].reset_index()

    timeline['time'] = timeline['month'] + "-" + timeline['year'].astype(str)

    return timeline

def daily_timeline(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    return df.groupby('only_date').count()['message'].reset_index()

def week_activity_map(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    return df['day_name'].value_counts()

def month_activity_map(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    return df['month'].value_counts()

def activity_heatmap(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    return df.pivot_table(index='day_name', columns='period', values='message', aggfunc='count').fillna(0)

def clean_chat(df):
    noise_patterns = ['image omitted', 'video omitted', 'audio omitted', 'document omitted']
    df = df[df['message'].notna()]  # Remove empty messages
    df = df[~df['message'].str.lower().str.contains('|'.join(noise_patterns))]
    return df

def analyze_sentiment_df(selected_user, df):
    # Keep sentiment analysis using the existing model
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    sentiment_results = []
    for index, row in df.iterrows():
        sentiment, confidence = analyzesenti(row['message'])
        sentiment_results.append({
            'user': row['user'],
            'message': row['message'],
            'Sentiment': sentiment,
            'Confidence': confidence
        })
    return pd.DataFrame(sentiment_results)
