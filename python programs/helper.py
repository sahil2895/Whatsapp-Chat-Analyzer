from urlextract import URLExtract
from wordcloud import WordCloud
import pandas as pd
from collections import Counter
import emoji
import openai
extractor= URLExtract()
from transformers import AutoTokenizer,AutoModelForSequenceClassification
from scipy.special import softmax

openai.api_key=" "
def predict_next_message(chat_context):
    prompt = f"Continue this conversation:\n{chat_context}\nUser:"
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=100
    )
    return response.choices[0].message.content.strip()

def generate_followup(chat_history):
    prompt = f"Generate a likely follow-up for this conversation:\n{chat_history}"
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=100
    )
    return response.choices[0].message.content.strip()

def detect_unusual_activity(messages):
    prompt = f"Identify any unusual patterns or spikes in activity in these messages:\n{messages}"
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=100
    )
    return response.choices[0].message.content.strip()

MODEL="j-hartmann/emotion-english-distilroberta-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

def analyzesenti(text,df):
    encoded_text = tokenizer(text[:512], return_tensors='pt')  
    output = model(**encoded_text)
    scores = output[0][0].detach().numpy()
    prob = softmax(scores)

    labels=model.config.id2label
    sentiment = labels[prob.argmax()]
    confidence = prob[prob.argmax()]

    return sentiment, confidence

def fetchstats(selected_user,df):
    if selected_user!="Overall":
        df=df[df['user']==selected_user]
    num_msg=df.shape[0]
    words=[]
    for message in df['message']:
        words.extend(message.split())
    num_img_msg = df[df['message'].str.contains('image omitted', case=False, na=False)].shape[0]

    words1=[]
    for message in df['message']:
        words1.extend(message.split())
    num_aud_msg = df[df['message'].str.contains('audio omitted', case=False, na=False)].shape[0]


    links=[]
    for message in df['message']:
        links.extend(extractor.find_urls(message))
    
    return num_msg , len(words), num_img_msg,num_aud_msg, len(links)

def busiest_users(df):
    busy=df['user'].value_counts().head()
    df = ((df['user'].value_counts() / df.shape[0]) * 100).round(2).reset_index()
    df.columns = ['name', 'percent']

    return busy,df

from wordcloud import WordCloud

def create_wordcloud(selected_user, df):

    with open('stop_hinglish.txt', 'r') as f:
        stop_words = set(f.read().split())

    if selected_user != "Overall":
        df = df[df['user'] == selected_user]

    temp = df[df['message'] != 'image omitted']

    if temp.empty:
        print("No valid messages available for word cloud.")
        return WordCloud(width=500, height=500, background_color="white").generate("No Data")

    def remove_stop_words(message):
        return " ".join(word for word in message.lower().split() if word not in stop_words)

    temp['message'] = temp['message'].apply(remove_stop_words)

    wc = WordCloud(width=500, height=500, min_font_size=10, background_color="white")
    df_wc = wc.generate(temp['message'].str.cat(sep=" "))

    return df_wc

def most_common_words(selected_user, df):

    with open('stop_hinglish.txt', 'r') as f:
        stop_words = set(f.read().split())

    if selected_user != "Overall":
        df = df[df['user'] == selected_user]

    temp = df[df['message'] != 'image omitted']

    words = []
    for message in temp['message']:
        for word in message.lower().split():
            if word not in stop_words:
                words.append(word)

    if not words:
        print("No valid words to display.")
        return pd.DataFrame(columns=["Word", "Frequency"])

    most_common_df = pd.DataFrame(Counter(words).most_common(20), columns=["Word", "Frequency"])

    return most_common_df

def emoji_helper(selected_user,df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    emojis = []
    for message in df['message']:
        emojis.extend([c for c in message if emoji.is_emoji(c)])

    emoji_df = pd.DataFrame(Counter(emojis).most_common(len(Counter(emojis))))

    return emoji_df

def monthly_timeline(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    timeline = df.groupby(['year', 'month_num', 'month']).count()['message'].reset_index()

    time = []
    for i in range(timeline.shape[0]):
        time.append(timeline['month'][i] + "-" + str(timeline['year'][i]))

    timeline['time'] = time

    return timeline

def daily_timeline(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    daily_timeline = df.groupby('only_date').count()['message'].reset_index()

    return daily_timeline

def week_activity_map(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    return df['day_name'].value_counts()

def month_activity_map(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    return df['month'].value_counts()

def activity_heatmap(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    user_heatmap = df.pivot_table(index='day_name', columns='period', values='message', aggfunc='count').fillna(0)

    return user_heatmap
