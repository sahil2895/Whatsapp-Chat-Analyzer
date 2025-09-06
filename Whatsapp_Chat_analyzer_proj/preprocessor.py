import re
import pandas as pd
import re
import pandas as pd

def preprocess(data):
    # Pattern to capture datetime like [06/09/25, 9:05:12 AM]
    pattern = r"\[(\d{2}/\d{2}/\d{2},\s\d{1,2}:\d{2}:\d{2}\s(?:AM|PM))\]\s"
    
    # Split data into messages and extract dates
    messages = re.split(pattern, data)[1:]  
    dates = messages[::2]   # every alternate element is a date
    messages = messages[1::2]  # actual messages
    
    # Build dataframe
    df = pd.DataFrame({'date': dates, 'user_msg': messages})
    df['date'] = pd.to_datetime(df['date'], format="%d/%m/%y, %I:%M:%S %p", errors="coerce")
    
    # Split user and message
    users, msgs = [], []
    for message in df['user_msg']:
        entry = re.split(r'([\w\W]+?):\s', message, maxsplit=1)
        if len(entry) > 2:
            users.append(entry[1])
            msgs.append(entry[2])
        else:
            users.append("group_notification")
            msgs.append(entry[0])
    
    df['user'] = users
    df['message'] = msgs
    df.drop(columns=['user_msg'], inplace=True)
    
    # Extract datetime features
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month_name()
    df['month_num'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['day_name'] = df['date'].dt.day_name()
    df['only_date'] = df['date'].dt.date
    df['hour'] = df['date'].dt.hour
    df['minute'] = df['date'].dt.minute
    df['period'] = df['hour'].apply(lambda x: f"{x}-{(x+1)%24}")
    
    return df

