import re
import pandas as pd
def preprocess(data):
    pattern = r"\[\d{2}/\d{2}/\d{2},\s\d{1,2}:\d{2}:\d{2}\u202f?(?:AM|PM)\]"
    messages=re.split(pattern,data)[1:]
    dates=re.findall(pattern,data)
    df = pd.DataFrame({'user_msg' : messages, 'msg_date': dates})

    df['msg_date'] = df['msg_date'].str.strip('[]')  # Remove square brackets

    df['msg_date'] = pd.to_datetime(df['msg_date'], format='%d/%m/%y, %I:%M:%S\u202f%p')

    df.rename(columns={'msg_date': 'date'}, inplace=True)
    users=[]
    messages=[]
    for message in df['user_msg']:
        entry=re.split(r'([\w\W]+?):\s',message)
        if entry[1:]:
            users.append(entry[1])
            messages.append(" ".join(entry[2:]))
        else:
            users.append('group_notification')
            messages.append(entry[0])

    df['user']=users
    df['message']=messages
    df.drop(columns=['user_msg'],inplace=True)
    df['year']=df['date'].dt.year
    df['month']=df['date'].dt.month_name()
    df['month_num']=df['date'].dt.month
    df['day']=df['date'].dt.day
    df['day_name']=df['date'].dt.day_name()
    df['only_date']=df['date'].dt.date
    df['hour']=df['date'].dt.hour
    df['minute']=df['date'].dt.minute
    df['period'] = df['hour'].apply(lambda x: f"{x}-{(x+1)%24}")
    return df
