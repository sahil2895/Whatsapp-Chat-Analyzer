import pandas as pd
import streamlit as st
import helper
import preprocessor
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(color_codes=True)

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd())))


st.sidebar.title("Whatsapp Chat Analyzer")

uploaded_file = st.sidebar.file_uploader("Choose a file")
if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    data = bytes_data.decode("utf-8")
    df = preprocessor.preprocess(data)
    st.dataframe(df)

    user_lst=df['user'].unique().tolist()

    user_lst.sort()
    user_lst.insert(0,"Overall")

    selected_user=st.sidebar.selectbox("Show analysis with respect to ",user_lst)
    if st.sidebar.button("Show Analysis"):
        num_msg, words, num_img_msg, num_aud_msg, num_links=helper.fetchstats(selected_user,df)

        column1, columnn2 , column3 ,column4 ,column5=st.columns(5)

        with column1:
            st.header("Total Number of Messages")
            st.title(num_msg)
        with columnn2:
            st.header("Total Number of words")
            st.title(words)
        with column3:
            st.header("Total Number of images")
            st.title(num_img_msg)

        with column4:
            st.header("Total Number of audios")
            st.title(num_aud_msg)

        with column5:
            st.header("Total Number of links")
            st.title(num_links)

        if selected_user=="Overall":
            st.title("Busiest Users")
            x, new_df=helper.busiest_users(df)
            fig=plt.subplot()

            column1,columnn2=st.columns(2)

            with column1:
                st.header("Busiest Users")
                fig = plt.figure()
                sns.barplot(x=x.index, y=x.values, color='cyan')
                plt.xlabel("Users")
                plt.ylabel("Frequency")
                plt.title("Busiest Users")
                plt.xticks(rotation='vertical')
                st.pyplot(fig)
            with columnn2:
                st.dataframe(new_df)

        df_wc=helper.create_wordcloud(selected_user,df)
        fig,ax= plt.subplots()
        ax.imshow(df_wc)
        st.title("WordCloud")
        st.pyplot(fig)

        most_common_df=helper.most_common_words(selected_user,df)
        st.dataframe(most_common_df)

        fig,ax=plt.subplots()
        ax.barh(most_common_df["Word"],most_common_df["Frequency"])
        plt.xticks(rotation='vertical')
        st.title('Most common words')
        st.pyplot(fig)

        emoji_df = helper.emoji_helper(selected_user,df)
        st.title("Emoji Analysis")

        col1,col2 = st.columns(2)

        with col1:
            st.dataframe(emoji_df)
        with col2:
            fig,ax = plt.subplots()
            # Use column names 'Count' and 'Emoji' instead of integer indices
            ax.pie(emoji_df['Count'].head(),labels=emoji_df['Emoji'].head(),autopct="%0.2f")
            st.pyplot(fig)

        st.title("Monthly Timeline")
        timeline = helper.monthly_timeline(selected_user,df)
        fig,ax = plt.subplots()
        ax.plot(timeline['time'], timeline['message'],color='green')
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

        st.title("Daily Timeline")
        daily_timeline = helper.daily_timeline(selected_user, df)
        fig, ax = plt.subplots()
        ax.plot(daily_timeline['only_date'], daily_timeline['message'], color='black')
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

        st.title('Activity Map')
        col1,col2 = st.columns(2)

        with col1:
            st.header("Most busy day")
            busy_day = helper.week_activity_map(selected_user,df)
            fig,ax = plt.subplots()
            ax.bar(busy_day.index,busy_day.values,color='purple')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        with col2:
            st.header("Most busy month")
            busy_month = helper.month_activity_map(selected_user, df)
            fig, ax = plt.subplots()
            ax.bar(busy_month.index, busy_month.values,color='orange')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        st.title("Weekly Activity Map")
        user_heatmap = helper.activity_heatmap(selected_user,df)
        fig,ax = plt.subplots()
        ax = sns.heatmap(user_heatmap)
        st.pyplot(fig)


        st.title("Sentiment Analysis")
        try:
            # Get sentiment analysis results using the new function
            sentiment_df = helper.analyze_sentiment_df(selected_user, df)

            st.write("### Sentiment Distribution")
            sentiment_counts = sentiment_df['Sentiment'].value_counts()

            # Create sentiment distribution chart
            fig, ax = plt.subplots()
            sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette='coolwarm', ax=ax)
            plt.xlabel("Sentiment")
            plt.ylabel("Count")
            plt.title("Sentiment Distribution")
            plt.xticks(rotation=45)
            st.pyplot(fig)

            # Display sentiment data
            st.write("### Sentiment Details")
            st.dataframe(sentiment_df[['user', 'message', 'Sentiment', 'Confidence']])

            # Add a pie chart of sentiment distribution
            fig, ax = plt.subplots()
            sentiment_df['Sentiment'].value_counts().plot.pie(autopct='%1.1f%%', ax=ax, colors=sns.color_palette('coolwarm', len(sentiment_df['Sentiment'].unique())))
            plt.title("Sentiment Distribution (Pie Chart)")
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Error in sentiment analysis: {e}")
            st.write("Could not perform sentiment analysis. Please check your data or try again.")
